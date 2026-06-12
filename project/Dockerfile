FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Модель всегда обучается внутри образа: pickle гарантированно совместим
# с версиями библиотек контейнера (seed зафиксирован — результат детерминирован);
# MLflow-трекинг в build не нужен
RUN DISABLE_MLFLOW=1 python -m src.models.train

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    LOG_LEVEL=INFO \
    MODEL_PATH=artifacts/model.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
