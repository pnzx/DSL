import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import scipy.sparse as sp
import os

class LightGCN(Model):
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
    user_test_dict = {}
    for user_item in test_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_test_dict:
            user_test_dict[user] = set()
        user_test_dict[user].add(item)
    
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    
    # 모든 사용자의 임베딩 계산
    user_emb, item_emb = model.light_gcn_propagate(adj_matrix)
    
    for user in user_test_dict:
        if user not in user_train_dict:
            continue
            
        # 사용자의 모든 아이템에 대한 점수 계산
        user_emb_batch = tf.gather(user_emb, [user])
        scores = tf.matmul(user_emb_batch, item_emb, transpose_b=True)
        scores = tf.squeeze(scores)
        
        # 학습 데이터의 아이템 제외
        scores = scores.numpy()
        scores[list(user_train_dict[user])] = float('-inf')
        
        # Top-K 아이템 추출
        max_k = max(k_list)
        top_items = np.argsort(scores)[-max_k:][::-1]
        
        # 평가 지표 계산
        test_items = user_test_dict[user]
        for k in k_list:
            top_k_items = set(top_items[:k])
            
            recall = len(top_k_items & test_items) / len(test_items)
            recalls[k].append(recall)
            
            dcg = 0
            idcg = 0
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(k, len(test_items))):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs[k].append(ndcg)
    
    results = {}
    for k in k_list:
        results[f'Recall@{k}'] = np.mean(recalls[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

def main():
    # 데이터 로드
    train_data = np.load('ml-1m_clean/train_list.npy')
    test_data = np.load('ml-1m_clean/test_list.npy')
    valid_data = np.load('ml-1m_clean/valid_list.npy')
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
    
    # 인접 행렬 생성
    adj_matrix = create_adj_matrix(train_data, num_users, num_items)
    
    # 하이퍼파라미터 수정
    EMBEDDING_DIM = 64
    NUM_LAYERS = 5  # 레이어 수 증가
    BATCH_SIZE = 4096  # 배치 크기 증가
    EPOCHS = 500  # 에포크 수 증가
    LEARNING_RATE = 0.01  # 학습률 증가
    NUM_NEG = 16  # 네거티브 샘플 수 증가
    
    # 모델과 옵티마이저 초기화
    model = LightGCN(num_users, num_items, EMBEDDING_DIM, NUM_LAYERS, item_emb)
    
    # 모델 빌드
    dummy_inputs = (
        tf.zeros((1,), dtype=tf.int32),
        tf.zeros((1,), dtype=tf.int32),
        tf.zeros((1,), dtype=tf.int32)
    )
    dummy_adj = tf.sparse.from_dense(tf.zeros((num_users + num_items, num_users + num_items)))
    _ = model((dummy_inputs), dummy_adj)
    
    # 학습률 스케줄러 수정
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
    
    # 학습
    num_batches = len(train_data) // BATCH_SIZE
    train_generator = get_train_batch(train_data, user_train_dict, num_items, BATCH_SIZE, num_neg=NUM_NEG)
    
    best_ndcg = 0
    best_epoch = 0
    patience = 20
    no_improve = 0
    
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
        
        # 평가
        metrics = evaluate(model, test_data, user_train_dict, adj_matrix, num_items)
        print(f'에포크 {epoch+1}/{EPOCHS}, 평균 손실: {avg_loss:.4f}')
        print('평가 결과:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
        
        # 모델 저장
        current_ndcg = metrics['NDCG@20']
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch + 1
            model.save_weights('lightgcn_best.weights.h5')
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 최종 평가
    print(f'\n최고 성능 모델 (에포크 {best_epoch}):')
    model.load_weights('lightgcn_best.weights.h5')
    final_metrics = evaluate(model, test_data, user_train_dict, adj_matrix, num_items)
    for metric, value in final_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()