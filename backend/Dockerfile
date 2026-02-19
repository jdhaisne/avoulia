# Backend RAG Chatbot (FastAPI + Haystack + Chroma)
# Chat (streaming SSE), documents (ingest/delete), RAG avec embeddings Foundry
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV CHROMA_PERSIST_DIR=/app/data/chroma
RUN mkdir -p /app/data/chroma

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
