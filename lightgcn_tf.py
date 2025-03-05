import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import scipy.sparse as sp
import os

class LightGCN(Model):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 초기 임베딩
        initializer = tf.keras.initializers.GlorotUniform()
        self.user_embedding = Embedding(num_users, embedding_dim,
                                     embeddings_initializer=initializer,
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.item_embedding = Embedding(num_items, embedding_dim,
                                     embeddings_initializer=initializer,
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        
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
        
        # 희소 텐서로 변환
        self.norm_adj = tf.sparse.from_sparse_tensor_value(
            tf.sparse.SparseTensor(
                indices=np.array([norm_adj.tocoo().row, norm_adj.tocoo().col]).T,
                values=norm_adj.tocoo().data,
                dense_shape=norm_adj.shape
            )
        )
    
    def call(self, inputs):
        user, pos_item, neg_item = inputs
        
        # 초기 임베딩
        users_emb = self.user_embedding(tf.range(self.num_users))
        items_emb = self.item_embedding(tf.range(self.num_items))
        all_emb = tf.concat([users_emb, items_emb], axis=0)
        embs = [all_emb]
        
        # 메시지 전파
        for layer in range(self.num_layers):
            all_emb = tf.sparse.sparse_dense_matmul(self.norm_adj, all_emb)
            embs.append(all_emb)
        
        # 다층 임베딩 결합
        embs = tf.stack(embs, axis=1)
        final_embs = tf.reduce_mean(embs, axis=1)
        
        # 사용자와 아이템 임베딩 분리
        users_emb_final, items_emb_final = tf.split(final_embs, [self.num_users, self.num_items])
        
        # 특정 사용자와 아이템의 임베딩 추출
        user_emb = tf.nn.embedding_lookup(users_emb_final, user)
        pos_item_emb = tf.nn.embedding_lookup(items_emb_final, pos_item)
        neg_item_emb = tf.nn.embedding_lookup(items_emb_final, neg_item)
        
        # 점수 계산
        pos_score = tf.reduce_sum(user_emb * pos_item_emb, axis=1)
        neg_score = tf.reduce_sum(user_emb * neg_item_emb, axis=1)
        
        return pos_score, neg_score, users_emb_final, items_emb_final

def get_train_batch(train_data, user_items, num_items, batch_size):
    num_samples = len(train_data)
    while True:
        # 랜덤하게 배치 인덱스 선택
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            neg_items = []
            
            # 네거티브 샘플링
            for user in users:
                while True:
                    neg_item = np.random.randint(0, num_items)
                    if neg_item not in user_items[user]:
                        neg_items.append(neg_item)
                        break
            
            yield users, pos_items, neg_items

@tf.function
def train_step(model, optimizer, users, pos_items, neg_items):
    with tf.GradientTape() as tape:
        pos_score, neg_score, users_emb, items_emb = model((users, pos_items, neg_items))
        loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_score - neg_score)))
        
        # L2 정규화
        reg_loss = tf.reduce_mean(tf.square(users_emb)) + tf.reduce_mean(tf.square(items_emb))
        loss += 1e-4 * reg_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def calculate_metrics(model, test_data, user_train_dict, num_items, k_list=[10, 20]):
    # 사용자별 테스트 아이템 딕셔너리 생성
    user_test_dict = {}
    for user_item in test_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_test_dict:
            user_test_dict[user] = set()
        user_test_dict[user].add(item)
    
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    
    # 각 사용자에 대해 평가
    for user in user_test_dict:
        if user not in user_train_dict or len(user_test_dict[user]) == 0:
            continue
            
        # 모든 아이템에 대한 점수 계산
        user_input = tf.convert_to_tensor([user] * num_items, dtype=tf.int32)
        item_input = tf.convert_to_tensor(range(num_items), dtype=tf.int32)
        neg_items = tf.zeros_like(item_input)  # 더미 네거티브 아이템
        
        # 점수 계산
        pos_score, _, _, _ = model((user_input, item_input, neg_items))
        scores = pos_score.numpy()
        
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
    test_data = np.load('ml-1m_clean/test_list.npy')
    
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
    
    # 모델과 옵티마이저 초기화
    model = LightGCN(num_users, num_items, EMBEDDING_DIM, NUM_LAYERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # 인접 행렬 생성
    model.create_adj_matrix(train_data)
    
    # 학습
    num_batches = len(train_data) // BATCH_SIZE
    train_generator = get_train_batch(train_data, user_train_dict, num_items, BATCH_SIZE)
    
    best_recall = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for _ in range(num_batches):
            users, pos_items, neg_items = next(train_generator)
            users = tf.convert_to_tensor(users, dtype=tf.int32)
            pos_items = tf.convert_to_tensor(pos_items, dtype=tf.int32)
            neg_items = tf.convert_to_tensor(neg_items, dtype=tf.int32)
            
            loss = train_step(model, optimizer, users, pos_items, neg_items)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        
        # 5 에포크마다 평가 수행
        if (epoch + 1) % 5 == 0:
            print(f'에포크 {epoch+1}/{EPOCHS}, 평균 손실: {avg_loss:.4f}')
            metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
            print('평가 결과:')
            for metric, value in metrics.items():
                print(f'{metric}: {value:.4f}')
            
            # 최고 성능 모델 저장
            if metrics['Recall@10'] > best_recall:
                best_recall = metrics['Recall@10']
                best_epoch = epoch + 1
                model.save_weights('lightgcn_model_best.weights.h5')
    
    print(f'\n최고 성능 모델 (에포크 {best_epoch}):')
    model.load_weights('lightgcn_model_best.weights.h5')
    final_metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
    for metric, value in final_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main() 