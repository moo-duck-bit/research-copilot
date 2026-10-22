FROM python:3.11-slim

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts
COPY eval ./eval

# root가 아닌 고정 UID로 실행 (k8s securityContext.runAsNonRoot와 맞춘다)
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && chown -R 10001:10001 /code
USER 10001

EXPOSE 8000

# keep-alive를 Ingress(Traefik)의 백엔드 연결 재사용 시간(90초)보다 길게 둔다.
# 서버가 먼저 닫으면, 프록시가 그 연결을 재사용하는 순간 요청이 끊길 수 있다.
# 종료 신호 뒤 열린 요청은 최대 10초만 기다리고, 나머지 시간은 백그라운드 분석 마무리에 쓴다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "100", "--timeout-graceful-shutdown", "10"]
