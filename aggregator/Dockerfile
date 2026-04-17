FROM python:3.10-slim
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY *.py .
 
ENV CHECKPOINT_DIR=/checkpoints
ENV NUM_ROUNDS=15
ENV MIN_CLIENTS=3
 
# Health check: verify gRPC port is listening
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost',8080)); s.close()" || exit 1
 
CMD ["python", "server.py"]
