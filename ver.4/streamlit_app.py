import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Hannanum
import joblib
import re
import os
import time
import base64

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 모델 및 데이터 로드
@st.cache_resource
def load_data():
    try:
        tfidf_vectorizer = joblib.load(os.path.join(current_dir, 'C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/tfidf_vectorizer.joblib'))
        cosine_sim = joblib.load(os.path.join(current_dir, 'C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/cosine_sim.joblib'))
        similar_genres = joblib.load(os.path.join(current_dir, 'C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/similar_genres.joblib'))
        webtoon_data = pd.read_pickle(os.path.join(current_dir, 'C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/processed_webtoon_data.pkl'))
        return tfidf_vectorizer, cosine_sim, similar_genres, webtoon_data
    except FileNotFoundError as e:
        st.error(f"필요한 파일을 찾을 수 없습니다: {e}")
        st.error("train_model.py 스크립트를 먼저 실행했는지 확인해주세요.")
        return None, None, None, None

tfidf_vectorizer, cosine_sim, similar_genres, webtoon_data = load_data()

if tfidf_vectorizer is None:
    st.stop()

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
    expanded_query = expand_query_genres(query)
    processed_query = preprocess_text(expanded_query)
    
    query_vec = tfidf_vectorizer.transform([processed_query])
    sim_scores = cosine_similarity(query_vec, tfidf_vectorizer.transform(webtoon_data['processed_features']))
    
    sim_scores = sim_scores.flatten()
    similar_webtoons = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
    
    return similar_webtoons[:3]  # 상위 3개만 반환

def recommend_webtoons(query):
    try:
        recommendations = get_recommendations(query)
        
        if not recommendations:
            return "조건에 맞는 웹툰을 찾지 못했습니다."
        
        response = "다음 웹툰들을 추천합니다:\n\n"
        for i, (idx, score) in enumerate(recommendations, 1):
            webtoon = webtoon_data.iloc[idx]
            response += f"{i}. {webtoon['title']} (유사도: {score:.2f})\n"
            response += f"   작가: {webtoon['author']}, 장르: {webtoon['genre']}\n"
            response += f"   연령 제한: {webtoon['age']}, "
            response += "완결" if webtoon['completed'] else "연재중"
            response += f"\n   줄거리: {webtoon['description'][:100]}...\n\n"
        
        response += "이 웹툰들이 마음에 드시나요? 다른 추천을 원하시면 새로운 질문을 해주세요."
        return response
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return "추천 과정에서 오류가 발생했습니다. 다른 질문을 해주시겠어요?"

# 아이콘 이미지를 Base64로 인코딩하는 함수
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 스타일 설정
def set_page_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f5;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
    }
    .stChatMessage.assistant {
        background-color: #E2E2E2;
    }
    </style>
    """, unsafe_allow_html=True)

# 커스텀 아이콘 설정
user_icon = get_image_base64("C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/1.jpg")
assistant_icon = get_image_base64("C:/Users/gadi2/OneDrive/바탕 화면/study file/240827_chatbot/ver.4/2.jpg")

# Streamlit UI
set_page_style()

st.title("웹툰 추천 챗봇")

# 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "안녕하세요! 웹툰 추천 챗봇입니다. 어떤 웹툰을 찾고 계신가요?",
        "avatar": f"data:image/png;base64,{assistant_icon}"
    })

# 종료 상태 확인
if "should_rerun" not in st.session_state:
    st.session_state.should_rerun = False
if "countdown" not in st.session_state:
    st.session_state.countdown = 0

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

# 카운트다운 표시
if st.session_state.countdown > 0:
    st.info(f"화면이 초기화되기까지 {st.session_state.countdown}초 남았습니다.")
    st.session_state.countdown -= 1
    time.sleep(1)
    st.rerun()

# 사용자 입력 처리
if st.session_state.countdown == 0:
    prompt = st.chat_input("원하는 웹툰에 대해 설명해주세요:")
    if prompt:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": f"data:image/png;base64,{user_icon}"
        })
        with st.chat_message("user", avatar=f"data:image/png;base64,{user_icon}"):
            st.markdown(prompt)

        # 종료 요청 확인
        if prompt.lower() in ['종료', '끝', '그만', 'quit', 'exit']:
            response = "웹툰 추천 서비스를 이용해 주셔서 감사합니다. 10초 후 화면이 초기화됩니다."
            st.session_state.should_rerun = True
            st.session_state.countdown = 10
        else:
            # 챗봇 응답 생성
            response = recommend_webtoons(prompt)

        # 챗봇 응답 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": f"data:image/png;base64,{assistant_icon}"
        })
        with st.chat_message("assistant", avatar=f"data:image/png;base64,{assistant_icon}"):
            st.markdown(response)

        st.rerun()

# 종료 후 페이지 새로고침
if st.session_state.should_rerun and st.session_state.countdown == 0:
    st.session_state.messages = []
    st.session_state.should_rerun = False
    st.rerun()

# 사이드바에 사용 방법 표시
st.sidebar.title("사용 방법")
st.sidebar.write("""
1. 원하는 웹툰의 특징을 입력창에 적어주세요.
2. 예: "학교를 배경으로 하는 로맨스 웹툰" 또는 "액션 판타지 웹툰"
3. 엔터를 누르면 챗봇이 웹툰을 추천해줍니다.
4. 추천된 웹툰 목록을 확인하세요!
5. 대화를 종료하려면 "종료", "끝", "그만", "quit", "exit",'바이','bye' 중 하나를 입력하세요.
""")