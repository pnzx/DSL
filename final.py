import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import scipy.sparse as sp
import os

class LightGCN(Model):
    """
    LightGCN 추천 모델 클래스
    
    Args:
        num_users (int): 전체 사용자 수
        num_items (int): 전체 아이템 수
        embedding_dim (int): 임베딩 차원 수
        num_layers (int): GCN 레이어 수
        item_embeddings (np.array, optional): 사전 학습된 아이템 임베딩
    """
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, item_embeddings=None):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # 사용자 임베딩 초기화 - 초기화 방식 수정
        self.user_embedding = Embedding(num_users, embedding_dim,
                                     embeddings_initializer=tf.keras.initializers.glorot_uniform(),
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-5))
        
        # 아이템 임베딩 초기화 (기존 임베딩 사용)
        if item_embeddings is not None:
            initializer = tf.keras.initializers.Constant(item_embeddings)
        else:
            initializer = tf.keras.initializers.glorot_uniform()
        self.item_embedding = Embedding(num_items, embedding_dim,
                                     embeddings_initializer=initializer,
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-5))
    
    def build(self, input_shape):
        # 임베딩 레이어 빌드
        self.user_embedding.build((None,))
        self.item_embedding.build((None,))
        self.built = True
    
    def light_gcn_propagate(self, adj_matrix):
        """
        LightGCN의 메시지 전파를 수행합니다.
        
        Args:
            adj_matrix (tf.sparse.SparseTensor): 정규화된 인접 행렬
        
        Returns:
            tuple: (user_embeddings, item_embeddings) 최종 임베딩
        """
        # 초기 임베딩 가져오기
        users_emb = tf.nn.embedding_lookup(self.user_embedding.embeddings, tf.range(self.num_users))
        items_emb = tf.nn.embedding_lookup(self.item_embedding.embeddings, tf.range(self.num_items))
        all_emb = tf.concat([users_emb, items_emb], axis=0)
        
        # 각 레이어의 임베딩 저장 (layer combination 가중치 적용)
        embs = [all_emb]
        layer_weights = [1.0/(self.num_layers + 1)] * (self.num_layers + 1)
        
        # 메시지 전파
        for layer in range(self.num_layers):
            all_emb = tf.sparse.sparse_dense_matmul(adj_matrix, all_emb)
            # L2 정규화 추가
            all_emb = tf.nn.l2_normalize(all_emb, axis=1)
            embs.append(all_emb)
        
        # 가중치가 적용된 레이어 결합
        all_embs = [embs[i] * layer_weights[i] for i in range(len(embs))]
        light_out = tf.add_n(all_embs)
        
        # 사용자와 아이템 임베딩 분리
        users, items = tf.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def call(self, inputs, adj_matrix):
        """
        모델의 순전파를 수행합니다.
        
        Args:
            inputs (tuple): (users, pos_items, neg_items) 배치 데이터
            adj_matrix (tf.sparse.SparseTensor): 정규화된 인접 행렬
        
        Returns:
            tuple: (positive_scores, negative_scores, user_embeddings, item_embeddings)
        """
        users, pos_items, neg_items = inputs
        
        # LightGCN 전파
        user_emb, item_emb = self.light_gcn_propagate(adj_matrix)
        
        # 배치에 해당하는 임베딩 추출
        user_emb_lookup = tf.nn.embedding_lookup(user_emb, users)
        pos_item_emb_lookup = tf.nn.embedding_lookup(item_emb, pos_items)
        neg_item_emb_lookup = tf.nn.embedding_lookup(item_emb, neg_items)
        
        # 점수 계산
        pos_scores = tf.reduce_sum(user_emb_lookup * pos_item_emb_lookup, axis=1)
        neg_scores = tf.reduce_sum(user_emb_lookup * neg_item_emb_lookup, axis=1)
        
        return pos_scores, neg_scores, user_emb, item_emb

def create_adj_matrix(train_data, num_users, num_items):
    """
    사용자-아이템 상호작용 데이터로부터 정규화된 인접 행렬을 생성합니다.
    
    Args:
        train_data (np.array): 사용자-아이템 상호작용 데이터 (user_id, item_id)
        num_users (int): 전체 사용자 수
        num_items (int): 전체 아이템 수
    
    Returns:
        tf.sparse.SparseTensor: 정규화된 인접 행렬
    """
    # 희소 행렬 생성
    adj = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    
    # 사용자-아이템 상호작용 기록
    for user, item in train_data:
        adj[user, num_users + item] = 1
        adj[num_users + item, user] = 1
    
    # 정규화
    rowsum = np.array(adj.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj).dot(d_mat)
    
    # Convert to COO format
    norm_adj = norm_adj.tocoo()
    
    # Convert to TensorFlow SparseTensor
    indices = np.column_stack((norm_adj.row, norm_adj.col))
    return tf.sparse.SparseTensor(
        indices=indices,
        values=norm_adj.data,
        dense_shape=norm_adj.shape
    )

def get_train_batch(train_data, user_items, num_items, batch_size, num_neg=16):
    """
    학습을 위한 미니배치를 생성하는 제너레이터 함수입니다.
    
    Args:
        train_data (np.array): 학습 데이터 (user_id, item_id)
        user_items (dict): 사용자별 상호작용한 아이템 집합
        num_items (int): 전체 아이템 수
        batch_size (int): 배치 크기
        num_neg (int): 각 positive 샘플당 생성할 negative 샘플 수
    
    Yields:
        tuple: (users, positive_items, negative_items) 배치 데이터
    """
    num_samples = len(train_data)
    # 아이템 인기도 계산
    item_popularity = np.zeros(num_items)
    for user, item in train_data:
        item_popularity[item] += 1
    item_popularity = item_popularity / np.sum(item_popularity)
    
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            neg_items = []
            
            # 개선된 네거티브 샘플링
            for user in users:
                user_negs = []
                user_pos_items = user_items[user]
                while len(user_negs) < num_neg:
                    # 인기도를 고려한 네거티브 샘플링
                    if np.random.random() < 0.5:  # 50% 확률로 인기도 기반 샘플링
                        neg_item = np.random.choice(num_items, p=item_popularity)
                    else:  # 50% 확률로 균등 샘플링
                        neg_item = np.random.randint(0, num_items)
                    if neg_item not in user_pos_items and neg_item not in user_negs:
                        user_negs.append(neg_item)
                neg_items.extend(user_negs)
            
            users = np.repeat(users, num_neg)
            pos_items = np.repeat(pos_items, num_neg)
            
            yield users, pos_items, neg_items

@tf.function
def train_step(model, optimizer, users, pos_items, neg_items, adj_matrix):
    """
    한 번의 학습 스텝을 수행합니다.
    
    Args:
        model (LightGCN): 학습할 모델
        optimizer (tf.keras.optimizers): 옵티마이저
        users (tf.Tensor): 사용자 ID 배치
        pos_items (tf.Tensor): 긍정적 아이템 ID 배치
        neg_items (tf.Tensor): 부정적 아이템 ID 배치
        adj_matrix (tf.sparse.SparseTensor): 정규화된 인접 행렬
    
    Returns:
        float: 현재 배치의 손실값
    """
    with tf.GradientTape() as tape:
        pos_scores, neg_scores, user_emb, item_emb = model((users, pos_items, neg_items), adj_matrix)
        
        # 개선된 BPR 손실 계산
        bpr_loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))
        
        # L2 정규화 손실
        l2_reg = 1e-5  # 정규화 계수 조정
        reg_loss = l2_reg * (tf.nn.l2_loss(user_emb) + tf.nn.l2_loss(item_emb))
        
        loss = bpr_loss + reg_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    # 그래디언트 클리핑 추가
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(model, test_data, user_train_dict, adj_matrix, num_items, k_list=[10, 20]):
    """
    순위 기반 모델 성능 평가
    
    Args:
        model (LightGCN): 평가할 모델
        test_data (np.array): 테스트 데이터 (user_id, item_id)
        user_train_dict (dict): 사용자별 학습 데이터의 아이템 집합
        adj_matrix (tf.sparse.SparseTensor): 정규화된 인접 행렬
        num_items (int): 전체 아이템 수
        k_list (list): 평가할 top-k 값들의 리스트
    
    Returns:
        dict: 각 평가 지표의 결과값 ('Recall@k', 'NDCG@k')
    """
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
    
    # 전체 임베딩 계산
    user_embeddings, item_embeddings = model.light_gcn_propagate(adj_matrix)
    
    # 각 사용자에 대해 평가
    for user in user_test_dict:
        if user not in user_train_dict or len(user_test_dict[user]) == 0:
            continue
        
        # 현재 사용자의 임베딩으로 모든 아이템에 대한 예측 점수 계산
        user_emb = tf.gather(user_embeddings, user)
        scores = tf.matmul(tf.expand_dims(user_emb, 0), item_embeddings, transpose_b=True)
        scores = tf.squeeze(scores)
        
        # 학습 데이터의 아이템은 제외 (이미 본 아이템)
        scores = scores.numpy()
        scores[list(user_train_dict[user])] = float('-inf')
        
        # 전체 아이템에 대한 순위 계산
        item_ranks = np.argsort(scores)[::-1]  # 점수 높은 순으로 정렬
        
        # 테스트 아이템들의 순위 찾기
        test_items = user_test_dict[user]
        max_k = max(k_list)
        top_items = item_ranks[:max_k]
        
        for k in k_list:
            top_k_items = set(item_ranks[:k])
            
            # Recall@K: 실제 본 아이템 중 상위 K개 안에 있는 비율
            recall = len(top_k_items & test_items) / len(test_items)
            recalls[k].append(recall)
            
            # NDCG@K: 순위 기반 평가
            # 1. 실제 테스트 아이템들의 순위 찾기
            test_item_ranks = []
            for item in test_items:
                rank = np.where(item_ranks == item)[0][0]
                if rank < k:  # 상위 K개 안에 있는 경우만 고려
                    test_item_ranks.append(rank + 1)  # 1-based ranking
            
            # 2. DCG 계산 (순위가 낮을수록 높은 가중치)
            dcg = 0
            for rank in test_item_ranks:
                dcg += 1 / np.log2(rank + 1)  # log2(rank + 1)로 순위에 따른 가중치 부여
            
            # 3. IDCG 계산 (이상적인 순서: 모든 테스트 아이템이 상위에 있는 경우)
            idcg = 0
            for i in range(min(k, len(test_items))):
                idcg += 1 / np.log2(i + 2)
            
            # 4. NDCG 계산
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs[k].append(ndcg)
    
    # 평균 계산
    results = {}
    for k in k_list:
        results[f'Recall@{k}'] = np.mean(recalls[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

def main():
    """
    LightGCN 모델의 학습과 평가를 위한 메인 함수
    """
    # 데이터 로드
    train_data = np.load('ml-1m_clean/train_list.npy')
    test_data = np.load('ml-1m_clean/test_list.npy')
    valid_data = np.load('ml-1m_clean/valid_list.npy')
    item_emb = np.load('ml-1m_clean/item_emb.npy')
    
    # 사용자/아이템 수 계산
    num_users = int(np.max(train_data[:, 0])) + 1
    num_items = int(np.max(train_data[:, 1])) + 1
    
    # 사용자별 아이템 기록
    user_train_dict = {}
    for user_item in train_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_train_dict:
            user_train_dict[user] = set()
        user_train_dict[user].add(item)
    
    # 그래프 생성
    adj_matrix = create_adj_matrix(train_data, num_users, num_items)
    
    # 하이퍼파라미터 설정
    EMBEDDING_DIM = 64
    NUM_LAYERS = 5
    BATCH_SIZE = 4096
    EPOCHS = 500
    LEARNING_RATE = 0.01
    NUM_NEG = 16
    EVAL_EVERY = 10      # 10 에포크마다 평가
    
    # 모델 설정
    model = LightGCN(num_users, num_items, EMBEDDING_DIM, NUM_LAYERS, item_emb)
    
    # 모델 초기화
    dummy_inputs = (
        tf.zeros((1,), dtype=tf.int32),
        tf.zeros((1,), dtype=tf.int32),
        tf.zeros((1,), dtype=tf.int32)
    )
    dummy_adj = tf.sparse.from_dense(tf.zeros((num_users + num_items, num_users + num_items)))
    _ = model((dummy_inputs), dummy_adj)
    
    # 학습률 스케줄러
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-4
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    # 학습 준비
    num_batches = len(train_data) // BATCH_SIZE
    train_generator = get_train_batch(train_data, user_train_dict, num_items, BATCH_SIZE, num_neg=NUM_NEG)
    
    # 학습 관련 변수들
    best_loss = float('inf')  # 최저 손실값 초기화
    best_epoch = 0
    patience = 20  # 얼리 스토핑 인내심
    no_improve = 0
    
    # 학습 시작
    for epoch in range(EPOCHS):
        total_loss = 0
        for _ in range(num_batches):
            users, pos_items, neg_items = next(train_generator)
            users = tf.convert_to_tensor(users, dtype=tf.int32)
            pos_items = tf.convert_to_tensor(pos_items, dtype=tf.int32)
            neg_items = tf.convert_to_tensor(neg_items, dtype=tf.int32)
            
            loss = train_step(model, optimizer, users, pos_items, neg_items, adj_matrix)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        print(f'에포크 {epoch+1}/{EPOCHS}, 평균 손실: {avg_loss:.4f}')
        
        # 평균 손실이 개선되었는지 확인
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            model.save_weights('lightgcn_best.weights.h5')
            no_improve = 0
        else:
            no_improve += 1
        
        # 10 에포크마다 검증 데이터로 평가
        if (epoch + 1) % EVAL_EVERY == 0:
            valid_metrics = evaluate(model, valid_data, user_train_dict, adj_matrix, num_items)
            print('검증 데이터 평가 결과:')
            for metric, value in valid_metrics.items():
                print(f'{metric}: {value:.4f}')
        
        # Early stopping 체크
        if no_improve >= patience:
            print(f'손실이 {patience}회 연속으로 개선되지 않아 {epoch+1}에포크에서 학습을 중단합니다.')
            break
    
    # 최종 테스트
    print(f'\n최고 성능 모델 (에포크 {best_epoch}, 최저 손실: {best_loss:.4f}):')
    model.load_weights('lightgcn_best.weights.h5')
    final_metrics = evaluate(model, test_data, user_train_dict, adj_matrix, num_items)
    print('테스트 데이터 최종 평가 결과:')
    for metric, value in final_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()