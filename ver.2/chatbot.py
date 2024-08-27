import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import joblib
import re
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 불용어 리스트
stop_words = [
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '라', '하다', '있다', '되다',
    '와인', '와', '의', '적', '전', '등', '임', '어', '가', '다', '또', '로', '수', '대', '용', '나', '것', '한', '모', '주', '어', '가', '주',
    '발효', '가스', '밸런스', '바디', '빈티지', '향', '피니시', '테이스팅', '스펙', '품종', '양조', '독일', '프랑스', '이탈리아', '스페인', '와인', '시음', '구입', '레드', '화이트', '로제', '디저트', '스파클링', '빈티지', '발효', '이탈리아', '프랑스', '스페인', '이탈리안', '프랑스어', '스페인어'
]

# 모델 및 데이터 로드
tfidf = joblib.load("wine_tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load("wine_tfidf_matrix.joblib")
w2v_model = Word2Vec.load("wine_w2v_model")
sentence_embeddings = joblib.load("wine_sentence_embeddings.joblib")
df = pd.read_pickle("wine_data.pkl")
word_to_ix = joblib.load("wine_word_to_ix.joblib")
label_to_ix = joblib.load("wine_label_to_ix.joblib")

# Sentence Transformer 모델 로드
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 딥러닝 모델 정의 및 로드
class WineEmbeddingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WineEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Max length 값을 설정해줘야 함
max_length = 50
deep_learning_model = WineEmbeddingModel(len(word_to_ix), 100, len(label_to_ix))
deep_learning_model.load_state_dict(torch.load("wine_deep_learning_model.pth"))
deep_learning_model.eval()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def pad_sequence(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [0] * (max_length - len(sequence))

def get_wine_recommendations(query, n=3):
    processed_query = preprocess_text(query)
    
    # TF-IDF 유사도
    query_tfidf = tfidf.transform([processed_query])
    tfidf_sims = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Word2Vec 유사도
    query_tokens = processed_query.split()
    if query_tokens:
        vectors = [w2v_model.wv[word] for word in query_tokens if word in w2v_model.wv]
        if vectors:
            query_vec = np.mean(vectors, axis=0)
            w2v_sims = cosine_similarity([query_vec], w2v_model.wv[range(len(w2v_model.wv))]).flatten()
        else:
            w2v_sims = np.zeros(tfidf_matrix.shape[0])
    else:
        w2v_sims = np.zeros(tfidf_matrix.shape[0])
    
    # Sentence Embedding 유사도
    query_embedding = sentence_model.encode([processed_query])
    sent_sims = cosine_similarity(query_embedding, sentence_embeddings).flatten()
    
    # 딥러닝 모델 예측
    indexed_query = [word_to_ix.get(word, 0) for word in processed_query.split()]
    padded_query = pad_sequence(indexed_query, max_length)
    with torch.no_grad():
        dl_output = deep_learning_model(torch.tensor([padded_query]))
    dl_sims = nn.functional.softmax(dl_output, dim=1).numpy().flatten()
    
    # 배열 크기를 TF-IDF와 맞추기
    def resize_array(array, size):
        if len(array) < size:
            return np.concatenate([array, np.zeros(size - len(array))])
        elif len(array) > size:
            return array[:size]
        return array

    size = tfidf_matrix.shape[0]
    
    w2v_sims = resize_array(w2v_sims, size)
    sent_sims = resize_array(sent_sims, size)
    dl_sims = resize_array(dl_sims, size)
    
    # 유사도 결합
    combined_sims = (tfidf_sims + w2v_sims + sent_sims + dl_sims) / 4
    
    # 키워드 기반 가중치 부여
    if any(keyword in query for keyword in ['저렴', '싼', '저가', '가성비']):
        combined_sims[df['가격대'].str.contains('저가')] *= 1.5
    elif any(keyword in query for keyword in ['비싼', '고가', '고급', '럭셔리']):
        combined_sims[df['가격대'].str.contains('고가')] *= 1.5
    
    if any(keyword in query for keyword in ['달콤', '단', '스위트']):
        combined_sims[df['단맛'] == '단맛'] *= 1.5
    elif any(keyword in query for keyword in ['쓴', '비터']):
        combined_sims[df['쓴맛'] == '쓴맛'] *= 1.5
    elif any(keyword in query for keyword in ['신맛', '산']):
        combined_sims[df['산미'] == '산미'] *= 1.5
    
    if any(keyword in query for keyword in ['고기', '쇠고기', '돼지고기', '닭고기']):
        combined_sims[df['어울리는 음식 종류'] == '육류'] *= 1.5
    elif any(keyword in query for keyword in ['생선', '새우', '오징어', '조개']):
        combined_sims[df['어울리는 음식 종류'] == '해산물'] *= 1.5
    elif any(keyword in query for keyword in ['야채', '채소', '샐러드']):
        combined_sims[df['어울리는 음식 종류'] == '채소'] *= 1.5
    elif any(keyword in query for keyword in ['디저트', '케이크', '초콜릿']):
        combined_sims[df['어울리는 음식 종류'] == '디저트'] *= 1.5
    
    if '특별한 경우' in df.columns:
        if any(keyword in query for keyword in ['기념일', '파티', '선물']):
            combined_sims[df['특별한 경우'] == '특별한 경우'] *= 1.5
    
    top_indices = combined_sims.argsort()[-n:][::-1]
    return df.iloc[top_indices]


def generate_response(query, recommendations):
    # 사용자의 질문에서 요구하는 정보 추출
    if '저렴하면서' in query:
        price_filter = '저가'
    else:
        price_filter = None

    if '파스타와 어울릴' in query:
        pairing_filter = '파스타'
    else:
        pairing_filter = None

    if '달콤한' in query:
        sweetness_filter = '단맛'
    else:
        sweetness_filter = None

    # 사용자 요청에 맞는 추천 와인 필터링
    filtered_recommendations = recommendations.copy()

    if price_filter:
        filtered_recommendations = filtered_recommendations[filtered_recommendations['가격대'].str.contains(price_filter)]

    if pairing_filter:
        filtered_recommendations = filtered_recommendations[filtered_recommendations['어울리는 음식 종류'].str.contains(pairing_filter)]

    if sweetness_filter:
        filtered_recommendations = filtered_recommendations[filtered_recommendations['단맛'] == sweetness_filter]

    response = f"'{query}'에 대한 와인 추천 결과입니다:\n\n"

    if filtered_recommendations.empty:
        response += "죄송하지만, 요청하신 조건에 맞는 와인을 찾을 수 없습니다."
    else:
        for i, (_, wine) in enumerate(filtered_recommendations.iterrows(), 1):
            response += f"{i}. {wine['와인 이름']} ({wine['원산지']}):\n"
            response += f"   - 특징: {wine['색상']} 와인, {wine['당분 함량']}한 맛, 탄산 {wine['탄산 유무']}, {wine['바디감']}\n"
            response += f"   - 가격대: {wine['가격대']}\n"
            response += f"   - 숙성 정도: {wine['숙성 정도']}\n"
            response += f"   - 양조 방법: {wine['양조 방법']}\n"
            response += f"   - 맛 특징: 쓴맛 {wine['쓴맛']}, 떫은맛 {wine['떫은맛']}, 단맛 {wine['단맛']}\n"
            response += f"   - 추천 페어링: {wine['어울리는 음식 종류']} ({wine['어울리는 음식 장르']})\n\n"

    response += "이 중에서 어떤 와인이 마음에 드시나요? 더 자세한 정보가 필요하면 말씀해 주세요."
    return response

def wine_chatbot(query):
    recommendations = get_wine_recommendations(query, n=3)
    response = generate_response(query, recommendations)
    return response

# 대화 루프
print("와인봇: 안녕하세요! 와인에 대해 궁금한 점을 물어보세요. ('종료'를 입력하면 대화가 종료됩니다.)")

while True:
    user_input = input("사용자: ")
    
    if user_input.lower() == '종료':
        print("와인봇: 대화를 종료합니다. 감사합니다!")
        break
    
    bot_response = wine_chatbot(user_input)
    print(f"와인봇: {bot_response}")
