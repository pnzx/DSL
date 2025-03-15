import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import os

class BPRMF(Model):
    """
    Bayesian Personalized Ranking Matrix Factorization (BPRMF) 모델 구현
    
    Args:
        num_users: 전체 사용자 수
        num_items: 전체 아이템 수
        embedding_dim: 임베딩 차원 수 (기본값: 64)
        item_embeddings: 사전 학습된 아이템 임베딩 (선택 사항)
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, item_embeddings=None):
        super(BPRMF, self).__init__()
        
        # 사용자 임베딩 - Xavier 초기화로 학습 안정성 높임
        self.user_embedding = Embedding(
            num_users, 
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.glorot_uniform(),
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
            name="user_embeddings"
        )
        
        # 아이템 임베딩 - 사전학습 임베딩 있으면 사용, 없으면 Xavier 초기화
        if item_embeddings is not None:
            # 외부에서 가져온 임베딩으로 초기화
            initializer = tf.keras.initializers.Constant(item_embeddings)
        else:
            # 없으면 그냥 Xavier로 초기화
            initializer = tf.keras.initializers.glorot_uniform()
            
        self.item_embedding = Embedding(
            num_items, 
            embedding_dim,
            embeddings_initializer=initializer,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
            name="item_embeddings"
        )
    
    def call(self, inputs):
        """
        모델의 순전파를 수행합니다.
        
        Args:
            inputs: (user, pos_item, neg_item) 배치 데이터 튜플
        
        Returns:
            (pos_score, neg_score): 긍정/부정 아이템의 예측 점수 튜플
        """
        user, pos_item, neg_item = inputs
        
        # 임베딩 추출
        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)
        
        # 내적으로 선호도 점수 계산 (유저와 아이템 벡터 간 유사도)
        pos_score = tf.reduce_sum(user_emb * pos_item_emb, axis=1)
        neg_score = tf.reduce_sum(user_emb * neg_item_emb, axis=1)
        
        return pos_score, neg_score

def get_train_batch(train_data, user_items, num_items, batch_size, num_neg=16):
    """
    학습을 위한 미니배치를 생성하는 제너레이터 함수입니다.
    
    Args:
        train_data: 학습 데이터 (user_id, item_id)
        user_items: 사용자별 상호작용한 아이템 집합
        num_items: 전체 아이템 수
        batch_size: 배치 크기
        num_neg: 각 positive 샘플당 생성할 negative 샘플 수
    
    Yields:
        (users, positive_items, negative_items): 배치 데이터 튜플
    """
    num_samples = len(train_data)
    
    # 아이템 인기도 계산 - 네거티브 샘플링에 활용할거임
    item_popularity = np.zeros(num_items)
    for user, item in train_data:
        item_popularity[item] += 1
    # 확률 분포로 변환 (총합=1)
    item_popularity = item_popularity / np.sum(item_popularity)
    
    # 무한 루프로 배치 계속 생성
    while True:
        # 데이터 셔플해서 학습 효과 높임
        indices = np.random.permutation(num_samples)
        
        # 배치 단위로 처리
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = train_data[batch_indices]
            
            users = batch_data[:, 0]
            pos_items = batch_data[:, 1]
            neg_items = []
            
            # 사용자별로 네거티브 샘플 뽑기
            for user in users:
                user_negs = []
                user_pos_items = user_items[user]  # 이 유저가 이미 본 아이템들
                
                # 필요한 만큼 네거티브 샘플 생성
                while len(user_negs) < num_neg:
                    # 인기도 기반 & 균등 샘플링 섞어서 사용 (다양성 확보)
                    if np.random.random() < 0.5:  # 50% 확률로 인기도 기반
                        neg_item = np.random.choice(num_items, p=item_popularity)
                    else:  # 나머지 50%는 완전 랜덤
                        neg_item = np.random.randint(0, num_items)
                        
                    # 이미 본 아이템이나 이미 뽑은 네거티브 아이템은 제외
                    if neg_item not in user_pos_items and neg_item not in user_negs:
                        user_negs.append(neg_item)
                
                neg_items.extend(user_negs)
            
            # 각 positive 샘플을 num_neg만큼 복제 (네거티브 샘플 수에 맞춤)
            users = np.repeat(users, num_neg)
            pos_items = np.repeat(pos_items, num_neg)
            
            yield users, pos_items, neg_items

@tf.function  # 그래프 모드로 실행하여 속도 향상
def train_step(model, optimizer, users, pos_items, neg_items):
    """
    한 번의 학습 스텝을 수행합니다.
    
    Args:
        model: 학습할 BPRMF 모델
        optimizer: 옵티마이저
        users: 사용자 ID 배치
        pos_items: 긍정적 아이템 ID 배치
        neg_items: 부정적 아이템 ID 배치
    
    Returns:
        현재 배치의 손실값
    """
    with tf.GradientTape() as tape:
        # 모델 순전파
        pos_score, neg_score = model((users, pos_items, neg_items))
        
        # BPR 손실 계산 - 긍정 아이템이 부정 아이템보다 점수가 높아야 함
        # log(sigmoid(pos_score - neg_score))를 최대화 = -log(sigmoid(pos_score - neg_score))를 최소화
        bpr_loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_score - neg_score))
        
        # L2 정규화 - 과적합 방지용
        l2_reg = 1e-5  # 정규화 강도
        reg_loss = l2_reg * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        
        # 전체 손실 = BPR 손실 + 정규화 손실
        total_loss = bpr_loss + reg_loss
        
    # 그래디언트 계산 및 적용
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # 그래디언트 클리핑 - 학습 안정화 (너무 큰 업데이트 방지)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    
    # 가중치 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

def calculate_metrics(model, test_data, user_train_dict, num_items, k_list=[10, 20]):
    """
    추천 모델의 성능을 평가합니다.
    
    Args:
        model: 평가할 BPRMF 모델
        test_data: 테스트 데이터 (user_id, item_id)
        user_train_dict: 사용자별 학습 데이터의 아이템 집합
        num_items: 전체 아이템 수
        k_list: 평가할 top-k 값들의 리스트
    
    Returns:
        각 평가 지표의 결과값 ('Recall@k', 'NDCG@k')
    """
    # 사용자별 테스트 아이템 딕셔너리 생성
    user_test_dict = {}
    for user_item in test_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_test_dict:
            user_test_dict[user] = set()
        user_test_dict[user].add(item)
    
    # 평가 지표 저장용 딕셔너리
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    
    # 각 사용자에 대해 평가
    for user in user_test_dict:
        # 학습 데이터나 테스트 데이터가 없는 사용자는 건너뜀
        if user not in user_train_dict or len(user_test_dict[user]) == 0:
            continue
            
        # 현재 사용자의 임베딩으로 모든 아이템에 대한 예측 점수 계산
        user_input = tf.convert_to_tensor([user] * num_items, dtype=tf.int32)
        item_input = tf.convert_to_tensor(range(num_items), dtype=tf.int32)
        
        user_emb = model.user_embedding(user_input)
        item_emb = model.item_embedding(item_input)
        scores = tf.reduce_sum(user_emb * item_emb, axis=1).numpy()
        
        # 학습 데이터의 아이템은 제외 (이미 본 아이템은 추천하면 안 됨)
        scores[list(user_train_dict[user])] = float('-inf')
        
        # 점수 기준으로 아이템 순위 계산
        item_ranks = np.argsort(scores)[::-1]  # 점수 높은 순으로 정렬
        
        # 테스트 아이템들
        test_items = user_test_dict[user]
        max_k = max(k_list)
        
        # 각 k값에 대해 평가 지표 계산
        for k in k_list:
            top_k_items = set(item_ranks[:k])
            
            # Recall@K: 실제 본 아이템 중 상위 K개 안에 있는 비율
            # 쉽게 말해 "맞춘 개수 / 전체 정답 개수"
            recall = len(top_k_items & test_items) / len(test_items)
            recalls[k].append(recall)
            
            # NDCG@K: 순위를 고려한 평가 지표 (순위가 높을수록 더 중요)
            # 1. 테스트 아이템들의 추천 순위 찾기
            test_item_ranks = []
            for item in test_items:
                rank = np.where(item_ranks == item)[0][0]
                if rank < k:  # 상위 K개 안에 있는 경우만 고려
                    test_item_ranks.append(rank + 1)  # 1-based ranking
            
            # 2. DCG 계산 (Discounted Cumulative Gain)
            # 순위가 낮을수록 가중치 감소
            dcg = 0
            for rank in test_item_ranks:
                dcg += 1 / np.log2(rank + 1)
            
            # 3. IDCG 계산 (Ideal DCG)
            # 이상적인 경우: 모든 테스트 아이템이 상위에 있을 때
            idcg = 0
            for i in range(min(k, len(test_items))):
                idcg += 1 / np.log2(i + 2)
            
            # 4. NDCG 계산 (DCG를 IDCG로 정규화)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs[k].append(ndcg)
    
    # 평균 계산하여 결과 반환
    results = {}
    for k in k_list:
        results[f'Recall@{k}'] = np.mean(recalls[k])
        results[f'NDCG@{k}'] = np.mean(ndcgs[k])
    
    return results

def main():
    """
    BPRMF 모델의 학습과 평가를 위한 메인 함수
    """
    print("BPRMF 추천 시스템 학습 시작!")
    
    print("데이터 로딩 중...")
    train_data = np.load('ml-1m_clean/train_list.npy')
    test_data = np.load('ml-1m_clean/test_list.npy')
    valid_data = np.load('ml-1m_clean/valid_list.npy')
    item_emb = np.load('ml-1m_clean/item_emb.npy')
    
    # 데이터 정보 출력
    print(f"- 학습 데이터: {len(train_data)}개 상호작용")
    print(f"- 검증 데이터: {len(valid_data)}개 상호작용")
    print(f"- 테스트 데이터: {len(test_data)}개 상호작용")
    
    # 사용자와 아이템 수 계산
    num_users = int(np.max(train_data[:, 0])) + 1
    num_items = int(np.max(train_data[:, 1])) + 1
    print(f"- 전체 사용자 수: {num_users}명")
    print(f"- 전체 아이템 수: {num_items}개")
    
    # 사용자별 아이템 목록 생성 (딕셔너리 형태)
    user_train_dict = {}
    for user_item in train_data:
        user = int(user_item[0])
        item = int(user_item[1])
        if user not in user_train_dict:
            user_train_dict[user] = set()
        user_train_dict[user].add(item)
    
    # 하이퍼파라미터 설정
    print("하이퍼파라미터 설정:")
    EMBEDDING_DIM = 64     
    BATCH_SIZE = 4096      
    EPOCHS = 500          
    LEARNING_RATE = 0.01   
    NUM_NEG = 16           
    EVAL_EVERY = 10        
    
    print(f"- 임베딩 차원: {EMBEDDING_DIM}")
    print(f"- 배치 크기: {BATCH_SIZE}")
    print(f"- 최대 에포크: {EPOCHS}")
    print(f"- 학습률: {LEARNING_RATE}")
    print(f"- 네거티브 샘플 수: {NUM_NEG}")
    
    # 모델과 옵티마이저 초기화
    print("BPRMF 모델 초기화 중...")
    model = BPRMF(num_users, num_items, EMBEDDING_DIM, item_emb)
    
    # 모델 초기화 - 더미 데이터로 첫 호출해서 가중치 초기화
    dummy_users = tf.zeros((1,), dtype=tf.int32)
    dummy_items = tf.zeros((1,), dtype=tf.int32)
    dummy_neg_items = tf.zeros((1,), dtype=tf.int32)
    _ = model((dummy_users, dummy_items, dummy_neg_items))
    print("- 모델 초기화 완료")
    
    # 학습률 스케줄러 - 코사인 감소 + 재시작 (학습 후반부 성능 향상)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-4
    )
    
    # AdamW 옵티마이저 - 가중치 감쇠 포함해서 일반화 성능 향상
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
    patience = 20  # 얼리 스토핑 인내심 (20번 연속 성능 향상 없으면 종료)
    no_improve = 0
    
    # 학습 시작
    print("\n모델 학습 시작...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in range(num_batches):
            # 배치 데이터 가져오기
            users, pos_items, neg_items = next(train_generator)
            users = tf.convert_to_tensor(users, dtype=tf.int32)
            pos_items = tf.convert_to_tensor(pos_items, dtype=tf.int32)
            neg_items = tf.convert_to_tensor(neg_items, dtype=tf.int32)
            
            # 학습 스텝 수행
            loss = train_step(model, optimizer, users, pos_items, neg_items)
            total_loss += loss
            
            # 진행 상황 표시 (10% 단위로 현황 보여주기)
            if (batch + 1) % (num_batches // 10) == 0:
                progress = (batch + 1) / num_batches * 100
                print(f"\r에포크 {epoch+1}/{EPOCHS} - {progress:.1f}% 완료 | 현재 손실: {loss:.4f}", end="")
        
        # 에포크 평균 손실 계산
        avg_loss = total_loss / num_batches
        print(f"\r에포크 {epoch+1}/{EPOCHS} 완료 | 평균 손실: {avg_loss:.4f}")
        
        # 평균 손실이 개선되었는지 확인
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        
        # 주기적으로 검증 데이터로 평가
        if (epoch + 1) % EVAL_EVERY == 0:
            print("\n검증 데이터 평가 중...")
            valid_metrics = calculate_metrics(model, valid_data, user_train_dict, num_items)
            print("검증 결과:")
            for metric, value in valid_metrics.items():
                print(f"  - {metric}: {value:.4f}")
            print()  # 줄바꿈
        
        # Early stopping 체크 - 성능 향상 없으면 일찍 종료
        if no_improve >= patience:
            print(f"손실이 {patience}회 연속으로 개선되지 않아 학습을 조기 종료합니다.")
            break
    
    # 최종 테스트
    print(f"\n최고 성능 모델 (에포크 {best_epoch}, 최저 손실: {best_loss:.4f})로 최종 평가 중...")
    final_metrics = calculate_metrics(model, test_data, user_train_dict, num_items)
    
    print("\n테스트 데이터 최종 평가 결과:")
    for metric, value in final_metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    print("\nBPRMF 모델 학습 및 평가 완료")

if __name__ == '__main__':
    main() 
    