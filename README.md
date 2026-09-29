# Langchain을 활용한 Chatbot 만들기
## 사용 기술
- python 3.11
- Langchain
- 환경 : pyenv 가상환경
- 화면 구성 : streamlit
- 배포 : steamlit cloud

## Streamlit 사용 이유
Streamlit은 기본적으로 React로 컴포넌트화하여 추상화 후 매핑해놓아 간단하게 화면 구성하기에 간편함
- 데이터 분석가가 시각화를 위해 자주 사용됨

## 배포시 주의 사항
패키지를 파일로 저장하여 함께 배포해야함
```
pip freeze > requirements.txt
```

