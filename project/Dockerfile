FROM python:3.11-slim

WORKDIR /app

# System deps (optional, mostly just to be safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: run the pipeline
CMD ["python", "-m", "src.pipeline.run_all", "--config", "config/config.yaml"]
