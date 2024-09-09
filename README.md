# 챗봇 개발 프로젝트: 데이터 기반 추천 시스템 개발 히스토리

## 프로젝트 개요

- **프로젝트명**: Create your own chatbot
- **기간**: 2024.8.27 ~ 8.28
- **목적**: 사용자 입력 기반 지능형 추천 시스템 개발
- **주요 도전 과제**: 제한된 데이터와 시간 내에서 효과적인 추천 모델 구축

## 기술 스택

- **프로그래밍 언어**: Python 3.8+
- **웹 프레임워크**: Streamlit 1.24.0
- **데이터 처리**: Pandas 1.3.3, Numpy 1.21.2
- **자연어 처리**: KoNLPy 0.5.2, scikit-learn 0.24.2 (TfidfVectorizer)
- **머신러닝/딥러닝**: PyTorch 1.9.0, Gensim 4.0.1 (Word2Vec), sentence-transformers 2.1.0
- **LLM 통합**: Langchain 0.0.184
- **배포 환경**: Localhost Streamlit 애플리케이션

## 상세 개발 과정 및 버전별 특징

### 버전 1: 와인 추천 챗봇 (초기 모델)

#### 개발 과정
1. **데이터 수집**:
   - Perplexity AI를 활용하여 50개의 와인 정보를 포함한 CSV 데이터셋 생성
   - 데이터 형식: `와인이름,종류,원산지,맛 특징,추천 음식`

2. **데이터 전처리**:
   ```python
   import pandas as pd
   
   df = pd.read_csv('wine_data.csv')
   wine_texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
   ```

3. **TF-IDF 모델 구축**:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   
   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(wine_texts)
   ```

4. **추천 로직 구현**:
   - 사용자 입력을 TF-IDF 벡터로 변환
   - 코사인 유사도를 사용하여 가장 유사한 와인 추천

#### 직면한 문제
- 제한된 데이터셋으로 인한 낮은 추천 정확도
- 사용자 질문의 다양성을 커버하기 어려움

#### 학습 포인트
- 품질 높은 대규모 데이터셋의 중요성 인식
- TF-IDF의 한계점 파악 (컨텍스트 이해 부족)

### 버전 2: RAG 기반 모델 개선 시도

#### 개발 과정
1. **데이터 확장**:
   - 와인 관련 질문-답변 쌍 100개 생성 (TXT 형식)

2. **고급 NLP 기법 적용**:
   - Word2Vec 모델 훈련:
     ```python
     from gensim.models import Word2Vec
     
     sentences = [text.split() for text in wine_texts]
     model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
     ```
   
   - RNN 모델 구현 (PyTorch 사용):
     ```python
     import torch.nn as nn
     
     class RNNModel(nn.Module):
         def __init__(self, vocab_size, embedding_dim, hidden_dim):
             super(RNNModel, self).__init__()
             self.embedding = nn.Embedding(vocab_size, embedding_dim)
             self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
             self.fc = nn.Linear(hidden_dim, vocab_size)
     
         def forward(self, x):
             embedded = self.embedding(x)
             output, _ = self.rnn(embedded)
             return self.fc(output)
     ```

#### 직면한 문제
- 복잡한 모델에 비해 여전히 부족한 데이터양
- RNN 모델의 과적합 문제
- 실제 대화 컨텍스트 반영의 어려움

#### 학습 포인트
- 데이터의 질과 양이 모델 복잡도보다 중요함을 인식
- 딥러닝 모델 훈련 시 과적합 방지의 중요성 경험

### 버전 3: LLM 통합 (Langchain 활용)

#### 개발 과정
1. **Langchain 설정**:
   ```python
   from langchain import OpenAI, LLMChain
   from langchain.prompts import PromptTemplate
   
   llm = OpenAI(model_name="gpt-4-mini")
   prompt = PromptTemplate(
       input_variables=["query"],
       template="와인 추천: {query}"
   )
   chain = LLMChain(llm=llm, prompt=prompt)
   ```

2. **데이터 통합**:
   - 버전 1의 CSV 데이터를 Langchain의 지식베이스로 활용

3. **추천 로직 구현**:
   ```python
   def get_recommendation(query):
       return chain.run(query)
   ```

#### 결과
- 놀라운 성능 향상: 다양한 질문에 대해 맥락을 이해하고 적절한 추천 제공

#### 학습 포인트
- LLM의 강력한 성능과 범용성 체감
- 기존 데이터셋의 효과적인 활용 방법 학습

### 버전 4: 웹툰 추천 시스템 (최종 버전)

#### 개발 과정
1. **데이터셋 변경**:
   - Kaggle의 네이버 웹툰 데이터셋 활용 (https://www.kaggle.com/datasets/bmofinnjake/naverwebtoon-datakorean)

2. **데이터 전처리**:
   ```python
   df = pd.read_csv('webtoon_data.csv')
   df['combined_info'] = df['author'] + ' ' + df['age_rating'] + ' ' + df['summary'] + ' ' + df['genre']
   ```

3. **TF-IDF 및 코사인 유사도 모델 구현**:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   
   tfidf = TfidfVectorizer()
   tfidf_matrix = tfidf.fit_transform(df['combined_info'])
   
   def get_recommendations(query):
       query_vec = tfidf.transform([query])
       similarities = cosine_similarity(query_vec, tfidf_matrix)
       similar_indices = similarities.argsort()[0][::-1][:5]
       return df.iloc[similar_indices]['title'].tolist()
   ```

4. **Streamlit 인터페이스 구현**:
   ```python
   import streamlit as st
   
   st.title("웹툰 추천 시스템")
   query = st.text_input("어떤 웹툰을 찾으시나요?")
   if query:
       recommendations = get_recommendations(query)
       st.write("추천 웹툰:", recommendations)
   ```

#### 결과
- 빠른 응답 속도
- 기본적인 키워드 기반 추천 가능, 그러나 깊이 있는 컨텍스트 이해 부족

#### 학습 포인트
- 도메인 특화 데이터의 중요성
- TF-IDF와 코사인 유사도의 실제 적용 경험
- Streamlit을 통한 빠른 프로토타이핑의 이점

## 주요 기술적 도전과 해결 방법

1. **데이터 부족 문제**:
   - 해결: 다양한 소스(Perplexity AI, Kaggle)에서 데이터 수집 및 생성
   - 교훈: 프로젝트 초기에 충분한 데이터 확보의 중요성

2. **모델 성능 개선**:
   - 시도: Word2Vec, RNN 등 고급 기법 적용
   - 결과: 데이터 양 대비 과도한 모델 복잡도로 인한 성능 저하
   - 해결: 최종적으로 간단하지만 효과적인 TF-IDF + 코사인 유사도 모델 채택

3. **사용자 인터페이스 구현**:
   - 해결: Streamlit을 사용한 간단하고 직관적인 웹 인터페이스 구현
   - 이점: 빠른 프로토타이핑과 사용자 피드백 수집 용이

## 향후 개선 방향

1. **데이터 확장**: 더 다양하고 풍부한 웹툰 데이터셋 구축
2. **고급 임베딩 기법 적용**: BERT 또는 최신 임베딩 모델 활용
3. **하이브리드 모델 개발**: TF-IDF 기반 모델과 딥러닝 모델의 앙상블
4. **사용자 피드백 시스템**: 추천 결과에 대한 사용자 평가 수집 및 모델 개선에 활용

## 결론

이 프로젝트를 통해 NLP와 추천 시스템 개발의 실제적인 도전 과제들을 경험했습니다. 특히 데이터의 중요성, 모델 복잡도와 성능 간의 균형, 그리고 사용자 중심 설계의 필요성을 깊이 있게 이해하게 되었습니다. 또한, 최신 LLM 기술의 강력함과 전통적인 ML 기법의 실용성을 비교 체험할 수 있었습니다. 이러한 경험은 향후 더 복잡하고 규모 있는 AI 프로젝트를 수행하는 데 있어 귀중한 토대가 될 것입니다.
