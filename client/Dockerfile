FROM python:3.10-slim
 
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*
 
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy application code
COPY *.py .
 
# The base model is downloaded at first run and cached in the volume
# Set HF cache to a volume-mountable path
ENV HF_HOME=/app/.cache/huggingface
 
CMD ["python", "client.py"]
