import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
from torch.utils.data import Dataset, DataLoader

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 초기 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 가중치 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def create_adj_matrix(self, train_data):
        # 사용자-아이템 인접 행렬 생성
        adj_mat = sp.dok_matrix((self.num_users + self.num_items,
                                self.num_users + self.num_items), dtype=np.float32)
        
        # 사용자-아이템 상호작용 추가
        for user_item in train_data:
            user = int(user_item[0])
            item = int(user_item[1])
            adj_mat[user, self.num_users + item] = 1.0
            adj_mat[self.num_users + item, user] = 1.0
        
        # 정규화된 인접 행렬 계산
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        # 희소 텐서로 변환 (수정된 부분)
        coo = norm_adj.tocoo()
        indices = np.array([coo.row, coo.col])
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(coo.data)
        self.norm_adj = torch.sparse_coo_tensor(
            indices, values, torch.Size(norm_adj.shape)
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, users, pos_items, neg_items=None):
        # 초기 임베딩
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        # 메시지 전파
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        
        # 다층 임베딩 결합
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        
        # 사용자와 아이템 임베딩 분리
        users_emb_final = final_embs[:self.num_users]
        items_emb_final = final_embs[self.num_users:]
        
        # 특정 사용자와 아이템의 임베딩 추출
        user_emb = users_emb_final[users]
        pos_item_emb = items_emb_final[pos_items]
        
        # 점수 계산
        pos_score = torch.sum(user_emb * pos_item_emb, dim=1)
        
        if neg_items is not None:
            neg_item_emb = items_emb_final[neg_items]
            neg_score = torch.sum(user_emb * neg_item_emb, dim=1)
            return pos_score, neg_score, users_emb_final, items_emb_final
        
        return pos_score, users_emb_final, items_emb_final

class TrainDataset(Dataset):
    def __init__(self, train_data, user_items, num_items):
        self.train_data = train_data
        self.user_items = user_items
        self.num_items = num_items
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        user = int(self.train_data[idx][0])
        pos_item = int(self.train_data[idx][1])
        
        # 네거티브 샘플링
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in self.user_items[user]:
                break
        
        return user, pos_item, neg_item

def train_step(model, optimizer, users, pos_items, neg_items):
    model.train()
    optimizer.zero_grad()
    
    pos_score, neg_score, users_emb, items_emb = model(users, pos_items, neg_items)
    
    # BPR 손실 계산
    loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))
    
    # L2 정규화
    reg_loss = 1e-4 * (torch.mean(torch.square(users_emb)) + torch.mean(torch.square(items_emb)))
    loss += reg_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def calculate_metrics(model, test_data, user_train_dict, num_items, k_list=[10, 20]):
    model.eval()
    
    # 사용자별 테스트 아이템 딕셔너리 생성
    user_test_dict = {}
    for user_item in test_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_test_dict:
            user_test_dict[user] = set()
        user_test_dict[user].add(item)
    
    device = next(model.parameters()).device
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    
    # 각 사용자에 대해 평가
    for user in user_test_dict:
        if user not in user_train_dict or len(user_test_dict[user]) == 0:
            continue
        
        # 모든 아이템에 대한 점수 계산
        user_input = torch.tensor([user], device=device).repeat(num_items)
        item_input = torch.arange(num_items, device=device)
        
        scores, _, _ = model(user_input, item_input)
        scores = scores.cpu().numpy()
        
        # 학습 데이터의 아이템은 제외
        scores[list(user_train_dict[user])] = float('-inf')
        
        # Top-K 아이템 추출
        max_k = max(k_list)
        top_items = np.argsort(scores)[-max_k:][::-1]
        
        # 지표 계산
        test_items = user_test_dict[user]
        for k in k_list:
            top_k_items = set(top_items[:k])
            # Recall@K
            recall = len(top_k_items & test_items) / len(test_items)
            recalls[k].append(recall)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(k, len(test_items))):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs[k].append(ndcg)
    
    # 평균 계산
    results = {}
    for k in k_list:
        results[f'Recall@{k}'] = np.mean(recalls[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

def main():
    # 데이터 로드
    train_data = np.load('ml-1m_clean/train_list.npy')
    valid_data = np.load('ml-1m_clean/valid_list.npy')
    test_data = np.load('ml-1m_clean/test_list.npy')
    item_emb = np.load('ml-1m_clean/item_emb.npy')
    
    # 사용자와 아이템 수 계산
    num_users = int(np.max(train_data[:, 0])) + 1
    num_items = int(np.max(train_data[:, 1])) + 1
    
    # 사용자별 아이템 목록 생성
    user_train_dict = {}
    for user_item in train_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_train_dict:
            user_train_dict[user] = set()
        user_train_dict[user].add(item)
    
    # 하이퍼파라미터 설정
    EMBEDDING_DIM = 64
    NUM_LAYERS = 3
    BATCH_SIZE = 1024
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'데이터셋 크기:')
    print(f'- 학습 데이터: {len(train_data)}')
    print(f'- 검증 데이터: {len(valid_data)}')
    print(f'- 테스트 데이터: {len(test_data)}')
    print(f'- 아이템 임베딩: {item_emb.shape}')
    print(f'학습 시작...\n')
    
    # 모델과 옵티마이저 초기화
    model = LightGCN(num_users, num_items, EMBEDDING_DIM, NUM_LAYERS).to(device)
    
    # 아이템 임베딩 초기화
    with torch.no_grad():
        model.item_embedding.weight.data.copy_(torch.FloatTensor(item_emb))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 데이터셋과 데이터로더 생성
    train_dataset = TrainDataset(train_data, user_train_dict, num_items)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f'인접 행렬 생성 중...')
    # 인접 행렬 생성
    model.create_adj_matrix(train_data)
    print(f'인접 행렬 생성 완료\n')
    
    best_recall = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f'에포크 {epoch+1}/{EPOCHS} 학습 중...')
        
        for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)  # shape: [batch_size, 4]
            
            loss = train_step(model, optimizer, users, pos_items, neg_items)
            total_loss += loss
            
            if (batch_idx + 1) % 100 == 0:
                print(f'배치 {batch_idx+1}/{len(train_loader)}, 손실: {loss:.4f}')
        
        avg_loss = total_loss / len(train_loader)
        
        # 5 에포크마다 평가 수행
        if (epoch + 1) % 5 == 0:
            print(f'\n에포크 {epoch+1}/{EPOCHS}, 평균 손실: {avg_loss:.4f}')
            
            # 검증 데이터로 평가
            print('검증 데이터로 평가 중...')
            valid_metrics = calculate_metrics(model, valid_data, user_train_dict, num_items)
            print('검증 결과:')
            for metric, value in valid_metrics.items():
                print(f'{metric}: {value:.4f}')
            
            # 최고 성능 모델 저장
            if valid_metrics['Recall@10'] > best_recall:
                best_recall = valid_metrics['Recall@10']
                best_epoch = epoch + 1
                torch.save(model.state_dict(), 'lightgcn_model_best.pt')
                print('최고 성능 모델 저장 완료')
        print('\n')
    
    print(f'\n최고 성능 모델 (에포크 {best_epoch}):')
    model.load_state_dict(torch.load('lightgcn_model_best.pt'))
    
    # 최종 테스트 데이터로 평가
    print('테스트 데이터로 최종 평가 중...')
    final_metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
    print('테스트 결과:')
    for metric, value in final_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()