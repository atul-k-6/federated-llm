import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes, ndarrays_to_parameters
import numpy as np
import torch, os, json, httpx
from pathlib import Path
 
from typing import Dict, List, Optional, Tuple, Union
 
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
EVAL_SERVICE   = os.environ.get("EVAL_SERVICE_URL", "http://eval:8000")
NUM_ROUNDS     = int(os.environ.get("NUM_ROUNDS", "15"))
 
 
class FedLoRAStrategy(FedAvg):
    """
    FedAvg strategy customized for LoRA weight averaging.
    Saves a checkpoint after each round and triggers evaluation.
    """
 
    def __init__(self, initial_parameters, **kwargs):
        super().__init__(
            initial_parameters=initial_parameters,
            **kwargs
        )
        self.round_results = []   # Track accuracy per round
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
 
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Override to add checkpoint saving after each round.
        The actual averaging is handled by the parent FedAvg class.
        """
        if not results:
            print(f"Round {server_round}: No results received!")
            return None, {}
 
        print(f"Round {server_round}: Aggregating {len(results)} client updates")
 
        # Log per-client training losses
        for client_proxy, fit_res in results:
            if fit_res.metrics:
                loss = fit_res.metrics.get("train_loss", "N/A")
                n    = fit_res.num_examples
                print(f"  Client: {n} examples, train_loss={loss:.4f}")
 
        # Run parent FedAvg averaging
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
 
        if aggregated_params is not None:
            # Save checkpoint as numpy arrays
            checkpoint_path = CHECKPOINT_DIR / f"round_{server_round:03d}.npz"
            arrays = fl.common.parameters_to_ndarrays(aggregated_params)
            np.savez(checkpoint_path, *arrays)
            print(f"  Checkpoint saved: {checkpoint_path}")
 
            # Signal eval service (non-blocking)
            self._trigger_evaluation(server_round)
 
        return aggregated_params, metrics
 
    def _trigger_evaluation(self, round_num: int):
        """POST to eval service to kick off accuracy measurement."""
        try:
            checkpoint_path = str(CHECKPOINT_DIR / f"round_{round_num:03d}.npz")
            resp = httpx.post(
                f"{EVAL_SERVICE}/evaluate",
                json={"round": round_num, "checkpoint": checkpoint_path},
                timeout=300.0,   # Evaluation can take several minutes
            )
            if resp.status_code == 200:
                result = resp.json()
                acc = result.get("accuracy", 0)
                print(f"  Eval round {round_num}: accuracy={acc:.4f}")
                self.round_results.append({
                    "round": round_num,
                    "accuracy": acc,
                    **result
                })
                # Save running results log
                with open(CHECKPOINT_DIR / "results.json", "w") as f:
                    json.dump(self.round_results, f, indent=2)
        except Exception as e:
            print(f"  Eval service error (round {round_num}): {e}")
 
 
def get_initial_parameters():
    """
    Get the initial LoRA weights before any training.
    Clients will receive these at round 0.
    """
    from model_utils import build_lora_model, get_lora_state_dict
    model = build_lora_model()
    lora_sd = get_lora_state_dict(model)
    arrays = [v.cpu().numpy() for v in lora_sd.values()]
    return ndarrays_to_parameters(arrays)
 
 
def main():
    min_clients = int(os.environ.get("MIN_CLIENTS", "1"))
    fraction_fit = float(os.environ.get("FRACTION_FIT", "1.0"))
 
    strategy = FedLoRAStrategy(
        initial_parameters=get_initial_parameters(),
        min_fit_clients=min_clients,        # Wait for all clients
        min_available_clients=min_clients,  # Before starting round
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,              # Disable client-side eval
    )
 
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
 
if __name__ == "__main__":
    main()
