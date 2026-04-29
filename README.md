# Federated Learning with Differential Privacy

This project simulates a Federated Learning (FL) environment using Docker, training a model on the CIFAR-10 dataset distributed across multiple clients, all coordinated by a central server using the FedAvg algorithm. It also includes Differential Privacy (DP) to add noise to the model updates.

## Quick Start Guide

Follow these steps to set up the environment, prepare the dataset, and run the simulation.

### 1. Set Up the Python Environment (For Windows)
You need a local Python environment to download and partition the dataset before starting the Docker containers.

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (If using CMD, run: .\venv\Scripts\activate.bat)

# Install required libraries (CPU only to save space/time)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Prepare the Dataset
The simulation requires the dataset to be physically partitioned into 3 distinct shards to enforce strict data privacy for each client.

```powershell
# Run the data preparation script
python prepare_dataset.py
```
This will download CIFAR-10, apply data augmentation, and split the training set into `data/client_0`, `data/client_1`, and `data/client_2`. It will also save the test set to `data/evaluator`.

### 3. Run the Federated Learning Simulation
Start the simulation using Docker Compose. You can control the amount of Differential Privacy noise injected into the model updates by passing the `NOISE_MULTIPLIER` environment variable.

**Run with the default noise (0.01):**
```powershell
docker compose up --build
```

**Run with a custom noise multiplier:**
```powershell
$env:NOISE_MULTIPLIER="0.05"; docker compose up --build
```

### 4. Cleaning Up (Before Re-running)
If you want to run the simulation again (e.g., with a different noise multiplier) and want a completely clean slate, clear the Docker caches and leftover volumes:

```powershell
# Remove old containers and volumes
docker compose down -v

# (Optional) Deep clean Docker cache
docker system prune -a -f
```

---

## Architecture & Hyperparameters

This section outlines all the properties of the model and the hyperparameters used across the client, server, and evaluator nodes in the federated learning setup.

### 1. Model Architecture
**Name:** `SimpleCNN` (located in `common/model.py`)
- **Conv1:** `nn.Conv2d(3, 32, 3, padding=1)` (3 input channels for RGB, 32 output channels, 3x3 kernel)
- **Activation:** `nn.ReLU()`
- **Pool:** `nn.MaxPool2d(2, 2)` (2x2 kernel, stride 2)
- **Conv2:** `nn.Conv2d(32, 64, 3, padding=1)` (32 input channels, 64 output channels, 3x3 kernel)
- **Flattening:** View reshaped to `(-1, 64 * 8 * 8)` (Input image is 32x32)
- **FC1:** `nn.Linear(64 * 8 * 8, 512)` (4096 input features, 512 output features)
- **Activation:** `nn.ReLU()`
- **FC2:** `nn.Linear(512, 10)` (512 input features, 10 output features for CIFAR-10 classes)

### 2. Training Hyperparameters (Client-Side)
Located in `client/client.py`:
- **Dataset:** CIFAR-10
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning Rate (LR):** `0.01`
- **Momentum:** `0.9`
- **Loss Function:** `CrossEntropyLoss`
- **Batch Size:** `32`
- **Local Epochs per Round:** `2`
- **Data Augmentation:** `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` applied to training data.
- **Data Partitioning:** Dataset is equally divided into 3 completely disjoint shards using the `prepare_dataset.py` script.

### 3. Differential Privacy (DP) Hyperparameters (Client-Side)
Located in `client/client.py`:
- **Noise Multiplier (`NOISE_MULTIPLIER`):** `0.01` (Default, dynamically configured via environment variables)
- **Clip Norm (`CLIP_NORM`):** `1.0`
- **Mechanism:** Global clipping of model weight deltas followed by adding Gaussian Noise `N(0, (NOISE_MULTIPLIER * CLIP_NORM)^2)`.

### 4. Federated Learning Hyperparameters (Server-Side)
Located in `server/main.py`:
- **Total Communication Rounds (`ROUNDS`):** `15`
- **Expected Clients (`EXPECTED_CLIENTS`):** `3`
- **Aggregation Strategy:** FedAvg-style delta aggregation (Global Model = Global Model + Average(Deltas from all clients))

### 5. Evaluator & Metrics Storage
Located in `evaluator/evaluate.py`:
- **Test Batch Size:** `64`
- **Evaluation Metrics (Saved per round):**
    - **Accuracy:** Overall performance on the CIFAR-10 test set.
    - **Test Loss:** Cross-entropy loss (indicates prediction confidence).
    - **Precision, Recall, F1-Score:** Macro-averaged metrics to detect class imbalances.
    - **Per-Class Accuracy:** Individual accuracy for each of the 10 CIFAR-10 classes.
- **Directory Structure:**
    - Metrics and checkpoints are organized into subdirectories based on the `NOISE_MULTIPLIER` to allow for easy comparison:
        - `metrics/{NOISE_MULTIPLIER}/round_{n}.json`
        - `checkpoints/{NOISE_MULTIPLIER}/client_{id}_round_{n}.pt`
