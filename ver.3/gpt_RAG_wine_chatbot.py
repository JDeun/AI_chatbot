import pandas as pd
import openai

# OpenAI API 키 설정
openai.api_key = '키를 입력하시면 됩니다'

# CSV 파일 로드
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# CSV 내용을 텍스트로 변환
def csv_to_text(df):
    text = ""
    for _, row in df.iterrows():
        text += " | ".join(map(str, row.values)) + "\n"
    return text

# GPT-4o-mini를 사용하여 질문에 대한 답변 생성
def generate_answer(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 와인 전문가입니다. 주어진 정보를 바탕으로 질문에 답해주세요."},
            {"role": "user", "content": f"다음은 관련 정보입니다:\n{context}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message['content']

# 메인 함수
def main():
    # 파일 경로를 적절한 형식으로 수정
    file_path = r'C:\Users\gadi2\OneDrive\바탕 화면\study file\240827_chatbot\wine_dataset.csv'  # CSV 파일 경로
    
    # CSV 파일을 로드하고 내용을 텍스트로 변환
    df = load_csv(file_path)
    context = csv_to_text(df)
    
    # 챗봇 소개 및 인사
    print("안녕하세요! 저는 와인 전문가 챗봇입니다.")
    print("와인에 대한 질문이 있으면 무엇이든 물어보세요. 종료하시려면 'exit' 또는 '종료'를 입력하세요.")
    
    while True:
        # 사용자 입력 받기
        question = input("질문을 입력하세요: ")
        
        # 종료 명령어 처리
        if question.lower() in ['exit', '종료']:
            print("프로그램을 종료합니다. 감사합니다!")
            break
        
        # 답변 생성 및 출력
        answer = generate_answer(question, context)
        print(f"답변: {answer}")

if __name__ == "__main__":
    main()
