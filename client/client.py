import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import io
from common.model import SimpleCNN

AGGREGATOR_URL = os.environ.get("AGGREGATOR_URL", "http://localhost:8000")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "3"))   

# DP parameters
NOISE_MULTIPLIER = float(os.environ.get("NOISE_MULTIPLIER", "0.01"))
CLIP_NORM = 1.0

def train(model, dataloader, epochs=2):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Client {CLIENT_ID} - Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

def add_dp_noise(deltas):
    # Calculate global norm of deltas
    total_norm = 0.0
    for key, tensor in deltas.items():
        total_norm += tensor.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = CLIP_NORM / (total_norm + 1e-6)
    clip_coef_clamped = min(1.0, clip_coef)
    
    noised_deltas = {}
    for key, tensor in deltas.items():
        # clip
        clipped_tensor = tensor * clip_coef_clamped
        # add Gaussian noise
        noise = torch.randn_like(clipped_tensor) * (NOISE_MULTIPLIER * CLIP_NORM)
        noised_deltas[key] = clipped_tensor + noise
        
    return noised_deltas

def main():
    print(f"Client {CLIENT_ID} starting...")
    
    # Load pre-partitioned dataset
    dataset_path = '/dataset/dataset.pt'
    print(f"Client {CLIENT_ID}: Loading pre-partitioned dataset from {dataset_path}...")
    
    # Check if dataset exists, since we depend on prepare_dataset.py having run
    while not os.path.exists(dataset_path):
        print(f"Waiting for dataset at {dataset_path}...")
        time.sleep(5)
        
    client_subset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    trainloader = torch.utils.data.DataLoader(client_subset, batch_size=32, shuffle=True)
    print(f"Client {CLIENT_ID} dataset size: {len(client_subset)}")
    
    model = SimpleCNN()
    
    current_round = 1
    total_rounds = 15
    
    os.makedirs(f'/app/checkpoints/{NOISE_MULTIPLIER}', exist_ok=True)
    
    while True:
        try:
            resp = requests.get(f"{AGGREGATOR_URL}/status")
            if resp.status_code == 200:
                status = resp.json()
                server_round = status["current_round"]
                total_rounds = status["total_rounds"]
                
                if server_round > total_rounds:
                    print(f"Client {CLIENT_ID}: Training complete.")
                    break
                    
                if server_round == current_round:
                    print(f"Client {CLIENT_ID}: Round {current_round} - Requesting model...")
                    resp = requests.get(f"{AGGREGATOR_URL}/get_model")
                    if resp.status_code == 200:
                        buffer = io.BytesIO(resp.content)
                        global_state = torch.load(buffer, map_location="cpu", weights_only=False)
                        model.load_state_dict(global_state)
                        
                        print(f"Client {CLIENT_ID}: Round {current_round} - Training...")
                        train(model, trainloader, epochs=2)
                        
                        # Save checkpoint
                        ckpt_path = f"/app/checkpoints/{NOISE_MULTIPLIER}/client_{CLIENT_ID}_round_{current_round}.pt"
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"Client {CLIENT_ID}: Saved checkpoint to {ckpt_path}")
                        
                        print(f"Client {CLIENT_ID}: Round {current_round} - Calculating deltas and adding DP noise...")
                        local_state = model.state_dict()
                        deltas = {k: local_state[k] - global_state[k] for k in local_state.keys()}
                        
                        noised_deltas = add_dp_noise(deltas)
                        
                        print(f"Client {CLIENT_ID}: Round {current_round} - Sending updates...")
                        out_buffer = io.BytesIO()
                        torch.save(noised_deltas, out_buffer)
                        out_buffer.seek(0)
                        
                        files = {'file': ('update.pt', out_buffer, 'application/octet-stream')}
                        requests.post(f"{AGGREGATOR_URL}/submit_update?client_id={CLIENT_ID}", files=files)
                        
                        current_round += 1
                else:
                    # Wait for other clients or next round
                    time.sleep(2)
        except Exception as e:
            print(f"Client {CLIENT_ID}: Waiting for server... ({e})")
            time.sleep(5)

if __name__ == "__main__":
    main()
