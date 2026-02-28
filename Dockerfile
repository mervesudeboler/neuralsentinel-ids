FROM python:3.11-slim

LABEL maintainer="mervesudeboler"
LABEL description="AdversaNet-IDS — Adversarial ML Intrusion Detection System"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# Create necessary directories
RUN mkdir -p data logs checkpoints reports

# Expose dashboard port
EXPOSE 8050

# Default: run full training then start dashboard
CMD ["sh", "-c", "python train.py && python src/dashboard/app.py"]
