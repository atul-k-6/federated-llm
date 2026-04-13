import torch, random, os
from pathlib import Path
 
def iid_shard(dataset, num_clients, output_dir):
    """
    Randomly shuffle and split into num_clients equal parts.
    Each client gets ~same class distribution as the full dataset.
    """
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
 
    shard_size = n // num_clients
    for client_id in range(num_clients):
        start = client_id * shard_size
        end   = start + shard_size if client_id < num_clients - 1 else n
        shard_indices = indices[start:end]
 
        shard = dataset.select(shard_indices)
        out_path = Path(output_dir) / f"client_{client_id + 1}"
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(shard, out_path / "data.pt")
        print(f"Client {client_id + 1}: {len(shard)} examples (IID)")

def non_iid_shard(dataset, num_clients, output_dir, classes_per_client=2):
    """
    Assign classes_per_client classes to each client.
    AG News has 4 classes (0-3). With 3 clients and 2 classes each:
      Client 1: classes 0, 1 (World, Sports)
      Client 2: classes 1, 2 (Sports, Business) -- some overlap
      Client 3: classes 2, 3 (Business, Tech)
    """
    # Group indices by class label
    class_indices = {}
    for idx, example in enumerate(dataset):
        label = example["label"].item()
        class_indices.setdefault(label, []).append(idx)
 
    all_classes = sorted(class_indices.keys())
 
    for client_id in range(num_clients):
        # Rotate which classes this client gets
        client_classes = [
            all_classes[(client_id + i) % len(all_classes)]
            for i in range(classes_per_client)
        ]
 
        # Collect indices for this client's classes
        client_indices = []
        for cls in client_classes:
            cls_idxs = class_indices[cls]
            # Each client gets a share of each of its classes
            share = len(cls_idxs) // (num_clients // classes_per_client + 1)
            start = (client_id // classes_per_client) * share
            client_indices.extend(cls_idxs[start:start + share])
 
        random.shuffle(client_indices)
        shard = dataset.select(client_indices)
        out_path = Path(output_dir) / f"client_{client_id + 1}"
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(shard, out_path / "data.pt")
        class_dist = {c: client_indices.count(i)
                      for c, idxs in class_indices.items()
                      for i in idxs if c in client_classes}
        print(f"Client {client_id + 1}: {len(shard)} examples, classes {client_classes}")

def create_validation_split(test_dataset, output_dir, val_fraction=0.15):
    """Reserve a held-out set that never goes to any client."""
    n = len(test_dataset)
    val_size = int(n * val_fraction)
    indices = list(range(n))
    random.shuffle(indices)
 
    val_data = test_dataset.select(indices[:val_size])
    out_path = Path(output_dir) / "validation"
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(val_data, out_path / "data.pt")
    print(f"Validation set: {len(val_data)} examples")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["iid", "non_iid"], default="non_iid")
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
 
    train = torch.load("data/processed/train_full.pt")
    test  = torch.load("data/processed/test_full.pt")
 
    if args.mode == "iid":
        iid_shard(train, args.num_clients, "data")
    else:
        non_iid_shard(train, args.num_clients, "data")
 
    create_validation_split(test, "data")
