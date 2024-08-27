import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 한국어 불용어 리스트 (예시, 필요에 따라 확장)
stop_words = ['은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '라']

def preprocess_text(text):
    # 특수 문자 제거 및 공백 정리
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 형태소 분석 및 불용어 제거
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# 데이터 로드 및 전처리
df = pd.read_csv('wine_dataset.csv')

# 텍스트 데이터 전처리
df['processed_text'] = df.apply(lambda row: preprocess_text(' '.join([
    str(row['와인 이름']), str(row['원산지']), str(row['탄산 유무']), str(row['색상']), 
    str(row['당분 함량']), str(row['양조 방법']), str(row['바디감']), str(row['숙성 정도']), 
    str(row['가격대']), str(row['쓴맛']), str(row['떫은맛']), str(row['단맛']), 
    str(row['어울리는 음식 종류']), str(row['어울리는 음식 장르'])
])), axis=1)

# Doc2Vec 모델 훈련
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(df['processed_text'])]
d2v_model = Doc2Vec(vector_size=100, min_count=2, epochs=30)
d2v_model.build_vocab(tagged_data)
d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

# TF-IDF 벡터화
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['processed_text'])

# 모델 및 데이터 저장
d2v_model.save("wine_d2v_model")
joblib.dump(tfidf, "wine_tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "wine_tfidf_matrix.joblib")
df.to_pickle("wine_data.pkl")

print("모델과 데이터가 성공적으로 저장되었습니다.")