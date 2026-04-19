import flwr as fl
import numpy as np
import torch
import os
 
from model import build_lora_model, get_lora_state_dict, set_lora_state_dict
from train import local_train
 
class FedLoRAClient(fl.client.NumPyClient):
    def __init__(self, client_id, dataset, config):
        self.client_id = client_id
        self.dataset   = dataset
        self.config    = config
        self.model     = build_lora_model()
        print(f"[Client {client_id}] Initialized with {len(dataset)} examples")
 
    def get_parameters(self, config):
        """Return LoRA weights as a flat list of NumPy arrays."""
        lora_sd = get_lora_state_dict(self.model)
        return [v.cpu().numpy() for v in lora_sd.values()]
 
    def set_parameters(self, parameters):
        """Load LoRA weights from aggregator into local model."""
        lora_sd = get_lora_state_dict(self.model)
        keys    = list(lora_sd.keys())
        state_dict = {
            k: torch.tensor(parameters[i])
            for i, k in enumerate(keys)
        }
        set_lora_state_dict(self.model, state_dict)
 
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model, avg_loss, epsilon = local_train(
            self.model, self.dataset, self.config
        )
        updated_params = self.get_parameters(config={})
        return updated_params, len(self.dataset), {
            "train_loss": avg_loss,
            "epsilon": epsilon,          # Report privacy budget to aggregator
        }
 
    def evaluate(self, parameters, config):
        """Quick local evaluation for Flower's built-in metrics."""
        self.set_parameters(parameters)
        self.model.eval()
        # (Eval service handles the real validation; this is lightweight)
        return 0.0, len(self.dataset), {"client_id": float(self.client_id)}
 
 
def main():
    client_id  = int(os.environ.get("CLIENT_ID", "1"))
    server_addr = os.environ.get("SERVER_ADDRESS", "aggregator:8080")
    data_path  = f"/data/data.pt"   # Volume-mounted path inside container
 
    dataset = torch.load(data_path)
 
    config = {
        "local_epochs": int(os.environ.get("LOCAL_EPOCHS", "2")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "16")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "2e-4")),
        "noise_multiplier": float(os.environ.get("NOISE_MULTIPLIER", "0.0")),
        "max_grad_norm": float(os.environ.get("MAX_GRAD_NORM", "1.0")),
        "target_delta": float(os.environ.get("TARGET_DELTA", "1e-5")),
    }
 
    client = FedLoRAClient(client_id, dataset, config)
 
    fl.client.start_numpy_client(
        server_address=server_addr,
        client=client,
    )
 
if __name__ == "__main__":
    main()
