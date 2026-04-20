# fastapi + rayserve


# uv 토치 설치
uv add torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match


# 다시 한번 뽑아보세요
uv export --format requirements-txt -o requirements.txt