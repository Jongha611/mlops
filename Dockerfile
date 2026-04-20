# 1. 베이스 이미지: 파이썬 3.12가 설치된 가벼운 데비안 리눅스
FROM python:3.12-slim

# 2. 환경 변수 설정: 파이썬이 .pyc 파일을 생성하지 않고, 로그를 즉시 출력하도록 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. 시스템 의존성 설치: 빌드에 필요한 도구들
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리 생성 및 설정
WORKDIR /app

# 5. 의존성 설치: 레이어 캐싱을 위해 requirements.txt만 먼저 복사
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 및 모델 복사
# (이때 models/ 폴더 내의 가중치 파일도 같이 포함됩니다)
COPY . .

# 7. 포트 개방: Ray Serve의 기본 포트인 8000번
EXPOSE 8000

# 8. 실행 명령: Ray Serve를 실행하여 배포 그래프를 띄움
# main.py에서 'deployment'라는 이름으로 bind 했다고 가정합니다.
CMD ["serve", "run", "main:deployment"]