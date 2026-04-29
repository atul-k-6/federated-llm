import os
import time
import json
import requests
import torch
import io
import matplotlib.pyplot as plt
from common.model import SimpleCNN

AGGREGATOR_URL = os.environ.get("AGGREGATOR_URL", "http://localhost:8000")
NOISE_MULTIPLIER = os.environ.get("NOISE_MULTIPLIER", "0.01")

def evaluate(model, dataloader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    correct = 0
    total = 0
    
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes
    
    with torch.no_grad():
        for data, target in dataloader:
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            for i in range(target.size(0)):
                label = target[i].item()
                pred = predicted[i].item()
                
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
                    true_positives[label] += 1
                else:
                    false_positives[pred] += 1
                    false_negatives[label] += 1
                    
    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    
    per_class_acc = {}
    precision_list = []
    recall_list = []
    f1_list = []
    
    # Class names for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    for i in range(num_classes):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        per_class_acc[classes[i]] = acc
        
        tp = true_positives[i]
        fp = false_positives[i]
        fn = false_negatives[i]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "per_class_accuracy": per_class_acc
    }

def main():
    print("Evaluator starting...")
    
    dataset_path = '/dataset/dataset.pt'
    print(f"Evaluator: Loading pre-downloaded dataset from {dataset_path}...")
    
    while not os.path.exists(dataset_path):
        print(f"Waiting for dataset at {dataset_path}...")
        time.sleep(5)
        
    testset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    model = SimpleCNN()
    
    accuracies = []
    rounds = []
    
    current_round = 1
    total_rounds = 15
    
    while True:
        try:
            resp = requests.get(f"{AGGREGATOR_URL}/status")
            if resp.status_code == 200:
                status = resp.json()
                server_round = status["current_round"]
                total_rounds = status["total_rounds"]
                
                if server_round > current_round:
                    print(f"Evaluating model after round {current_round}...")
                    model_resp = requests.get(f"{AGGREGATOR_URL}/get_model")
                    if model_resp.status_code == 200:
                        buffer = io.BytesIO(model_resp.content)
                        global_state = torch.load(buffer, map_location="cpu", weights_only=False)
                        model.load_state_dict(global_state)
                        
                        metrics = evaluate(model, testloader)
                        print(f"After Round {current_round}: Accuracy: {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}")
                        accuracies.append(metrics['accuracy'])
                        rounds.append(current_round)
                        
                        metrics_dir = f"/app/metrics/{NOISE_MULTIPLIER}"
                        os.makedirs(metrics_dir, exist_ok=True)
                        metrics_path = os.path.join(metrics_dir, f"round_{current_round}.json")
                        
                        # Include metadata like noise multiplier
                        metrics_to_save = {
                            "round": current_round,
                            "noise_multiplier": float(NOISE_MULTIPLIER),
                            **metrics
                        }
                        
                        with open(metrics_path, "w") as f:
                            json.dump(metrics_to_save, f, indent=4)
                        
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
    plt.savefig(f"accuracy_plot_{NOISE_MULTIPLIER}.png")
    print(f"Plot saved as accuracy_plot_{NOISE_MULTIPLIER}.png")

if __name__ == "__main__":
    main()
