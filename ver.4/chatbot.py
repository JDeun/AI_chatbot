# chatbot.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Hannanum
import joblib
import re

# 모델 및 데이터 로드
tfidf_vectorizer = joblib.load('ver.4/tfidf_vectorizer.joblib')
cosine_sim = joblib.load('ver.4/cosine_sim.joblib')
similar_genres = joblib.load('ver.4/similar_genres.joblib')
webtoon_data = pd.read_pickle('ver.4/processed_webtoon_data.pkl')

print(f"불러온 데이터 열: {webtoon_data.columns.tolist()}")

hannanum = Hannanum()

# 불용어 정의
stop_words = set(['것', '등', '들', '및', '을', '를', '이', '가'])
stop_words |= set("를 의 웹 툰 웹툰 웹툰판 이번 들 등 수 이 부 판 뿐 여자 남자 그 것 나 그 그녀 속 시작 속".split())
stop_words |= set("이야기 데 전 후 두 앞 뒤 그들 때문 사람 두 신작 한 자신 소년 소녀 만화".split())

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    nouns = hannanum.nouns(text)
    filtered_nouns = [word for word in nouns if word not in stop_words]
    return ' '.join(filtered_nouns)

def expand_query_genres(query):
    expanded_query = query
    for genre, similar in similar_genres.items():
        if genre in query.lower():
            expanded_query += ' ' + ' '.join(similar)
    return expanded_query

def get_recommendations(query, cosine_sim=cosine_sim):
    # 쿼리 확장 및 전처리
    expanded_query = expand_query_genres(query)
    processed_query = preprocess_text(expanded_query)
    
    # 쿼리를 TF-IDF 벡터로 변환
    query_vec = tfidf_vectorizer.transform([processed_query])
    
    # 쿼리와 모든 웹툰 사이의 유사도 계산
    sim_scores = cosine_similarity(query_vec, tfidf_vectorizer.transform(webtoon_data['processed_features']))
    
    # 유사도에 따라 웹툰 정렬
    sim_scores = sim_scores.flatten()
    similar_webtoons = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
    
    # 상위 5개 추천 웹툰 반환
    return similar_webtoons[:5]

def recommend_webtoons(query):
    try:
        recommendations = get_recommendations(query)
        
        if not recommendations:
            return "죄송합니다. 조건에 맞는 웹툰을 찾지 못했습니다."
        
        response = "다음 웹툰들을 추천합니다:\n\n"
        for i, (idx, score) in enumerate(recommendations, 1):
            webtoon = webtoon_data.iloc[idx]
            response += f"{i}. {webtoon['title']} (유사도: {score:.2f})\n"
            response += f"   작가: {webtoon['author']}, 장르: {webtoon['genre']}\n"
            response += f"   연령 제한: {webtoon['age']}, "
            response += "완결" if webtoon['completed'] else "연재중"
            response += f"\n   줄거리: {webtoon['description'][:100]}...\n\n"
        
        return response
    except Exception as e:
        print(f"Error occurred: {e}")
        return "죄송합니다. 추천 과정에서 오류가 발생했습니다. 다른 질문을 해주시겠어요?"

def chat():
    print("안녕하세요! 웹툰 추천 챗봇입니다. 질문을 입력하세요.")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ['exit', '종료', 'quit']:
            print("챗봇 종료. 안녕히 가세요!")
            break
        
        response = recommend_webtoons(user_input)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    chat()