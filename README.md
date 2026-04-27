# Federated LoRA (Low-RAM Run Guide)

This project can run with different numbers of clients using Docker Compose profiles.

## Why this helps on 8 GB RAM

- Each client loads and trains a DistilBERT+LoRA model.
- Running 3 clients in parallel can saturate CPU and memory.
- By default, Compose now starts only `aggregator` + `client_1`.

## Recommended startup commands

### 1 client (safest)

```bash
NOISE_MULTIPLIER=0.0 MIN_CLIENTS=1 docker compose up --build
```

### 2 clients

```bash
NOISE_MULTIPLIER=0.0 MIN_CLIENTS=2 \
docker compose --profile multi-client up --build
```

### 3 clients

```bash
NOISE_MULTIPLIER=0.0 MIN_CLIENTS=3 \
docker compose --profile multi-client --profile full-mesh up --build
```

## Extra knobs for low-memory machines

These defaults are already lowered in `docker-compose.yml`:

- `BATCH_SIZE=8`
- `LOCAL_EPOCHS=1`
- `TORCH_NUM_THREADS=1`
- `TORCH_NUM_INTEROP_THREADS=1`

You can make training lighter with:

```bash
BATCH_SIZE=4 LOCAL_EPOCHS=1 TORCH_NUM_THREADS=1 TORCH_NUM_INTEROP_THREADS=1 \
NOISE_MULTIPLIER=0.0 MIN_CLIENTS=1 docker compose up
```

