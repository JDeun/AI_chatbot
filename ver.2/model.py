import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import re
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 불용어 리스트
stop_words = [
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만', '라', '하다', '있다', '되다',  # 일반 불용어
    '와인', '와', '의', '적', '전', '등', '임', '어', '가', '다', '또', '로', '수', '대', '용', '나', '것', '한', '모', '주', '어', '가', '주',  # 와인 관련 불용어
    '발효', '가스', '밸런스', '바디', '빈티지', '향', '피니시', '테이스팅', '스펙', '품종', '양조', '독일', '프랑스', '이탈리아', '스페인', '와인', '시음', '구입', '레드', '화이트', '로제', '디저트', '스파클링', '빈티지', '발효', '이탈리아', '프랑스', '스페인', '이탈리안', '프랑스어', '스페인어'
]

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = okt.morphs(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# 데이터 로드 및 전처리
df = pd.read_csv('wine_dataset.csv')
df['processed_text'] = df.apply(lambda row: preprocess_text(' '.join([
    str(row['와인 이름']), str(row['원산지']), str(row['탄산 유무']), str(row['색상']), 
    str(row['당분 함량']), str(row['양조 방법']), str(row['바디감']), str(row['숙성 정도']), 
    str(row['가격대']), str(row['쓴맛']), str(row['떫은맛']), str(row['단맛']), 
    str(row['어울리는 음식 종류']), str(row['어울리는 음식 장르'])
])), axis=1)

# TF-IDF 벡터화
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['processed_text'])

# Word2Vec 모델 훈련
sentences = [text.split() for text in df['processed_text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Sentence Transformer
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
sentence_embeddings = sentence_model.encode(df['processed_text'].tolist())

# 딥러닝 모델 정의
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

# 데이터셋 및 DataLoader 정의
class WineDataset(Dataset):
    def __init__(self, texts, labels, word_to_ix, max_length):
        self.texts = [self.pad_sequence([word_to_ix.get(word, 0) for word in text.split()], max_length) for text in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    @staticmethod
    def pad_sequence(sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [0] * (max_length - len(sequence))

# 단어 사전 생성
word_to_ix = {word: i+1 for i, word in enumerate(set(' '.join(df['processed_text']).split()))}
word_to_ix['<PAD>'] = 0

# 레이블 생성 (예: 와인 유형을 레이블로 사용)
label_to_ix = {label: i for i, label in enumerate(df['색상'].unique())}
labels = [label_to_ix[label] for label in df['색상']]

# 데이터셋 및 DataLoader 생성
max_length = 100  # 적절한 max_length 설정
dataset = WineDataset(df['processed_text'], labels, word_to_ix, max_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화 및 훈련
model = WineEmbeddingModel(len(word_to_ix), 100, len(label_to_ix))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 200
for epoch in range(num_epochs):
    for batch_texts, batch_labels in dataloader:
        optimizer.zero_grad()
        output = model(batch_texts)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} completed")

# 모델 및 데이터 저장
joblib.dump(tfidf, "wine_tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "wine_tfidf_matrix.joblib")
w2v_model.save("wine_w2v_model")
joblib.dump(sentence_embeddings, "wine_sentence_embeddings.joblib")
torch.save(model.state_dict(), "wine_deep_learning_model.pth")
joblib.dump(word_to_ix, "wine_word_to_ix.joblib")
joblib.dump(label_to_ix, "wine_label_to_ix.joblib")
df.to_pickle("wine_data.pkl")

print("모델과 데이터가 성공적으로 저장되었습니다.")