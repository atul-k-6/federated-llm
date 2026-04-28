import os
import torch
import torchvision
import torchvision.transforms as transforms

def main():
    print("Preparing CIFAR-10 dataset...")
    
    # Training transform with Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize for 3-channel RGB
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Testing transform (NO Augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download raw datasets
    trainset = torchvision.datasets.CIFAR10(root='./data_raw', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data_raw', train=False, download=True, transform=test_transform)
    
    # Create output directories
    os.makedirs('./data/client_0', exist_ok=True)
    os.makedirs('./data/client_1', exist_ok=True)
    os.makedirs('./data/client_2', exist_ok=True)
    os.makedirs('./data/evaluator', exist_ok=True)
    
    # Split the trainset among 3 clients
    dataset_size = len(trainset)
    num_clients = 3
    shard_size = dataset_size // num_clients
    indices = list(range(dataset_size))
    
    for i in range(num_clients):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size
        client_indices = indices[start_idx:end_idx]
        client_subset = torch.utils.data.Subset(trainset, client_indices)
        
        # Save using torch.save
        torch.save(client_subset, f'./data/client_{i}/dataset.pt')
        print(f"Saved dataset for Client {i}: {len(client_subset)} samples.")
        
    # Save the full testset for the evaluator
    torch.save(testset, './data/evaluator/dataset.pt')
    print(f"Saved dataset for Evaluator: {len(testset)} samples.")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
