from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import torch
import io
import os
import asyncio

app = FastAPI()

ROUNDS = 15
EXPECTED_CLIENTS = 3
NOISE_MULTIPLIER = os.environ.get("NOISE_MULTIPLIER", "0.01")

current_round = 1
global_model_state = None
client_updates = []

@app.on_event("startup")
def startup_event():
    global global_model_state
    from common.model import SimpleCNN
    model = SimpleCNN()
    # Initialize global model
    global_model_state = model.state_dict()
    print("Aggregator started. Initialized global model.")

@app.get("/get_model")
def get_model():
    """Returns the latest global model weights."""
    global global_model_state
    buffer = io.BytesIO()
    torch.save(global_model_state, buffer)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")

@app.get("/status")
def status():
    """Returns the current training status."""
    return {"current_round": current_round, "total_rounds": ROUNDS}

@app.post("/submit_update")
async def submit_update(client_id: int, file: UploadFile = File(...)):
    """Receives weight deltas from a client, aggregates them when all received."""
    global client_updates, current_round, global_model_state
    
    if current_round > ROUNDS:
        return {"status": "done"}
        
    contents = await file.read()
    buffer = io.BytesIO(contents)
    delta_state = torch.load(buffer, map_location="cpu")
    
    client_updates.append(delta_state)
    print(f"Received update from client {client_id} for round {current_round}. Total updates: {len(client_updates)}")
    
    if len(client_updates) == EXPECTED_CLIENTS:
        print(f"Aggregating updates for round {current_round}...")
        # Aggregate deltas: global_state = global_state + average(deltas)
        new_state = {}
        for key in global_model_state.keys():
            avg_delta = sum([delta[key] for delta in client_updates]) / EXPECTED_CLIENTS
            new_state[key] = global_model_state[key] + avg_delta
            
        global_model_state = new_state
        client_updates = []
        print(f"Round {current_round} complete.")
        
        if current_round == ROUNDS:
            print(f"Federated training complete. Saving final global model for noise {NOISE_MULTIPLIER}...")
            torch.save(global_model_state, f"final_global_model_{NOISE_MULTIPLIER}.pt")
            
        current_round += 1
        
    return {"status": "received"}
