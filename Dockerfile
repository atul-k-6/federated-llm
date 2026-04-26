FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MNIST to /dataset so it's baked into the image and not overwritten by volumes
RUN python -c "import torchvision; torchvision.datasets.MNIST(root='/dataset', train=True, download=True); torchvision.datasets.MNIST(root='/dataset', train=False, download=True)"

COPY . .
