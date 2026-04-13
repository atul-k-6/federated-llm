from datasets import load_dataset
from transformers import AutoTokenizer
import torch, os
 
def download_and_tokenize(output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
 
    # Download AG News (auto-cached in ~/.cache/huggingface)
    print("Downloading AG News...")
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]
    test_data  = dataset["test"]
 
    # Tokenize with DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
 
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,    # Keep short for CPU training speed
            padding="max_length",
        )
 
    train_tokenized = train_data.map(tokenize_fn, batched=True, batch_size=1000)
    test_tokenized  = test_data.map(tokenize_fn, batched=True, batch_size=1000)
 
    # Keep only the columns we need
    cols = ["input_ids", "attention_mask", "label"]
    train_tokenized = train_tokenized.select_columns(cols)
    test_tokenized  = test_tokenized.select_columns(cols)
 
    # Set PyTorch format
    train_tokenized.set_format(type="torch", columns=cols)
    test_tokenized.set_format(type="torch", columns=cols)
 
    # Save
    torch.save(train_tokenized, f"{output_dir}/train_full.pt")
    torch.save(test_tokenized,  f"{output_dir}/test_full.pt")
    print(f"Saved {len(train_tokenized)} train, {len(test_tokenized)} test examples")
    return train_tokenized, test_tokenized
 
if __name__ == "__main__":
    download_and_tokenize()
