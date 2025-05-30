# 베이스 이미지
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y fonts-nanum

# Streamlit 앱 복사
COPY . .

# Streamlit 기본 설정 (CLI 설정)
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ENABLECORS false
ENV STREAMLIT_SERVER_PORT 8501

# Streamlit 실행 명령
CMD ["streamlit", "run", "standard_viewer.py"]
