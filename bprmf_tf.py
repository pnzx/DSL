import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import os

class BPRMF(Model):
    def __init__(self, num_users, num_items, embedding_dim=64, item_embeddings=None):
        super(BPRMF, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_dim,
                                     embeddings_initializer='random_normal',
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        
        # 아이템 임베딩 초기화
        initializer = tf.keras.initializers.Constant(item_embeddings) if item_embeddings is not None else 'random_normal'
        self.item_embedding = Embedding(num_items, embedding_dim,
                                     embeddings_initializer=initializer,
                                     embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
    
    def call(self, inputs):
        user, pos_item, neg_item = inputs
        
        # 임베딩 추출
        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)
        
        # 점수 계산
        pos_score = tf.reduce_sum(user_emb * pos_item_emb, axis=1)
        neg_score = tf.reduce_sum(user_emb * neg_item_emb, axis=1)
        
        return pos_score, neg_score

def get_train_batch(train_data, user_items, num_items, batch_size, num_neg=5):
    num_samples = len(train_data)
    while True:
        # 랜덤하게 배치 인덱스 선택
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            
            users = np.repeat(batch_data[:, 0], num_neg)
            pos_items = np.repeat(batch_data[:, 1], num_neg)
            neg_items = []
            
            # 네거티브 샘플링 개선
            for user in batch_data[:, 0]:
                user_negs = []
                while len(user_negs) < num_neg:
                    neg_item = np.random.randint(0, num_items)
                    if neg_item not in user_items[user] and neg_item not in user_negs:
                        user_negs.append(neg_item)
                neg_items.extend(user_negs)
            
            yield users, pos_items, neg_items

@tf.function
def train_step(model, optimizer, users, pos_items, neg_items):
    with tf.GradientTape() as tape:
        pos_score, neg_score = model((users, pos_items, neg_items))
        loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_score - neg_score)))
        # L2 정규화 추가
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss += 0.00001 * l2_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    # 그래디언트 클리핑 추가
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
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
        
        # 임베딩 추출
        user_emb = model.user_embedding(user_input)
        item_emb = model.item_embedding(item_input)
        
        # 점수 계산
        scores = tf.reduce_sum(user_emb * item_emb, axis=1)
        
        # 학습 데이터에 있는 아이템은 제외
        scores = scores.numpy()
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
    item_emb = np.load('ml-1m_clean/item_emb.npy')
    valid_data = np.load('ml-1m_clean/valid_list.npy')
    
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
    
    # 하이퍼파라미터 수정
    EMBEDDING_DIM = 64     # 임베딩 차원 유지
    BATCH_SIZE = 256      # 배치 크기 더 감소
    EPOCHS = 100          # 에포크 수 유지
    LEARNING_RATE = 0.0001  # 학습률 더 감소
    NUM_NEG = 8           # 네거티브 샘플 수 증가
    
    # 모델과 옵티마이저 초기화
    model = BPRMF(num_users, num_items, EMBEDDING_DIM, item_emb)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # 학습
    num_batches = len(train_data) // BATCH_SIZE
    train_generator = get_train_batch(train_data, user_train_dict, num_items, BATCH_SIZE, num_neg=NUM_NEG)
    
    best_ndcg = 0
    best_epoch = 0
    patience = 10  # Early stopping 용 patience
    no_improve = 0
    
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
        
        # 매 에포크마다 평가 수행
        metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
        print(f'에포크 {epoch+1}/{EPOCHS}, 평균 손실: {avg_loss:.4f}')
        print('평가 결과:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
        
        # NDCG@20 기준으로 모델 저장
        current_ndcg = metrics['NDCG@20']
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch + 1
            model.save_weights('bprmf_model_best.weights.h5')
            no_improve = 0
        else:
            no_improve += 1
            
        # Early stopping
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    print(f'\n최고 성능 모델 (에포크 {best_epoch}):')
    model.load_weights('bprmf_model_best.weights.h5')
    final_metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
    for metric, value in final_metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main() 
    