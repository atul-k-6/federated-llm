import os
import time
import requests
import torch
import torchvision
import torchvision.transforms as transforms
import io
import matplotlib.pyplot as plt
from common.model import SimpleCNN

AGGREGATOR_URL = os.environ.get("AGGREGATOR_URL", "http://localhost:8000")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def main():
    print("Evaluator starting...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Evaluator: Loading dataset from /dataset...")
    data_dir = '/dataset'
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    model = SimpleCNN()
    
    accuracies = []
    rounds = []
    
    current_round = 1
    total_rounds = 5
    
    while True:
        try:
            resp = requests.get(f"{AGGREGATOR_URL}/status")
            if resp.status_code == 200:
                status = resp.json()
                server_round = status["current_round"]
                total_rounds = status["total_rounds"]
                
                if server_round > current_round:
                    # A round has finished! The server has updated the model.
                    print(f"Evaluating model after round {current_round}...")
                    model_resp = requests.get(f"{AGGREGATOR_URL}/get_model")
                    if model_resp.status_code == 200:
                        buffer = io.BytesIO(model_resp.content)
                        global_state = torch.load(buffer, map_location="cpu")
                        model.load_state_dict(global_state)
                        
                        acc = evaluate(model, testloader)
                        print(f"After Round {current_round} Accuracy: {acc:.2f}%")
                        accuracies.append(acc)
                        rounds.append(current_round)
                        
                        current_round += 1
                        
                if current_round > total_rounds:
                    print("Federated training complete. Plotting results...")
                    break
            time.sleep(5)
        except Exception as e:
            print("Evaluator waiting for server...")
            time.sleep(5)
            
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b')
    plt.title("Global Model Accuracy vs Communication Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    # Save the plot in the current directory so it maps out of the container
    plt.savefig("accuracy_plot.png")
    print("Plot saved as accuracy_plot.png")

if __name__ == "__main__":
    main()
