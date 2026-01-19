# Builder
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN useradd -m llmcord && chown -R llmcord /app
USER llmcord

CMD ["python", "llmcord.py"]
