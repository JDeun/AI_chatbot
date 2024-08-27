import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 한국어 불용어 리스트 (예시, 필요에 따라 확장)
stop_words = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '라']

# 모델 및 데이터 로드
d2v_model = Doc2Vec.load("wine_d2v_model")
tfidf = joblib.load("wine_tfidf_vectorizer.joblib")
tfidf_matrix = joblib.load("wine_tfidf_matrix.joblib")
df = pd.read_pickle("wine_data.pkl")

def preprocess_text(text):
    # 특수 문자 제거 및 공백 정리
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 형태소 분석 및 불용어 제거
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def get_wine_recommendations(query, n=3):
    processed_query = preprocess_text(query)
    
    # Doc2Vec 유사도
    query_vector = d2v_model.infer_vector(processed_query.split())
    d2v_sims = d2v_model.dv.most_similar([query_vector], topn=len(df))
    
    # TF-IDF 유사도
    query_tfidf = tfidf.transform([processed_query])
    tfidf_sims = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # 결과 결합
    combined_sims = []
    for i, (idx, score) in enumerate(d2v_sims):
        idx = int(idx)
        if idx < len(tfidf_sims):
            combined_sims.append((idx, (score + tfidf_sims[idx]) / 2))
    
    top_indices = sorted(combined_sims, key=lambda x: x[1], reverse=True)[:n]
    
    return df.iloc[[idx for idx, _ in top_indices]]

def filter_recommendations(recommendations, filters):
    for column, value in filters.items():
        if column == '가격대':
            recommendations = recommendations[recommendations[column].str.contains(value, case=False)]
        elif column in ['원산지', '색상']:
            recommendations = recommendations[recommendations[column] == value]
        elif column in ['당분 함량', '탄산 유무', '바디감']:
            recommendations = recommendations[recommendations[column].str.contains(value, case=False)]
    return recommendations

def generate_wine_details(wine):
    details = f"{wine['와인 이름']} ({wine['원산지']}):\n"
    details += f"  {wine['색상']} 와인, {wine['당분 함량']}한 맛, 탄산 {wine['탄산 유무']}, {wine['바디감']}, 가격대 {wine['가격대']}\n"
    details += f"  숙성 정도: {wine['숙성 정도']}, 양조 방법: {wine['양조 방법']}\n"
    details += f"  맛 특징: 쓴맛 {wine['쓴맛']}, 떫은맛 {wine['떫은맛']}, 단맛 {wine['단맛']}\n"
    details += f"  추천 페어링: {wine['어울리는 음식 종류']} ({wine['어울리는 음식 장르']})"
    return details

def wine_chatbot(query):
    filters = {}
    
    # 키워드 매칭
    if re.search(r'한국|아시안|퓨전', query):
        filters['어울리는 음식 장르'] = '아시안|퓨전'
    
    if re.search(r'달콤|단', query):
        filters['당분 함량'] = '달콤|약간 달콤'
    
    if re.search(r'저렴|싼|가격이 낮은|가격대가 낮은', query):
        filters['가격대'] = '저가|중가'
    
    if '탄산' in query:
        filters['탄산 유무'] = '있음'
    
    if '화이트' in query:
        filters['색상'] = '화이트'
    elif '레드' in query:
        filters['색상'] = '레드'
    
    if re.search(r'가벼운|라이트', query):
        filters['바디감'] = '라이트|라이트-미디엄'
    elif re.search(r'무거운|풀', query):
        filters['바디감'] = '미디엄-풀|풀'
    
    countries = ['프랑스', '이탈리아', '스페인', '미국', '칠레', '아르헨티나', '호주', '뉴질랜드', '독일', '오스트리아']
    for country in countries:
        if country in query:
            filters['원산지'] = country
            break

    recommendations = get_wine_recommendations(query, n=20)
    filtered_recommendations = filter_recommendations(recommendations, filters)

    if len(filtered_recommendations) > 3:
        filtered_recommendations = filtered_recommendations.sample(n=3)
    elif len(filtered_recommendations) == 0:
        return "죄송합니다. 조건에 맞는 와인을 찾지 못했습니다. 다른 조건으로 다시 시도해 주세요."

    response = f"'{query}'에 대한 추천 와인입니다:\n\n"
    for _, wine in filtered_recommendations.iterrows():
        response += generate_wine_details(wine) + "\n\n"
    
    return response.strip()

# 대화 루프
print("와인봇: 안녕하세요! 와인에 대해 궁금한 점을 물어보세요. ('종료'를 입력하면 대화가 종료됩니다.)")

while True:
    user_input = input("사용자: ")
    
    if user_input == '종료':
        print("와인봇: 대화를 종료합니다. 감사합니다!")
        break
    
    bot_response = wine_chatbot(user_input)
    print(f"와인봇: {bot_response}")