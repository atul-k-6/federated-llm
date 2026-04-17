import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
 
def local_train(model, dataset, config):
    """
    Train model on local data for config["local_epochs"] epochs.
    Returns: (trained model, avg training loss)
    """
    model.train()
    device = torch.device("cpu")  # CPU for free-tier compatibility
    model.to(device)
 
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=0,  # Must be 0 inside Docker on some systems
    )
 
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 2e-4),
        weight_decay=0.01,
    )
 
    total_steps = len(loader) * config.get("local_epochs", 2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
 
    total_loss = 0.0
    num_steps = 0
 
    for epoch in range(config.get("local_epochs", 2)):
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
 
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
 
            loss = outputs.loss
            loss.backward()
 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
 
            total_loss += loss.item()
            num_steps += 1
 
    avg_loss = total_loss / num_steps
    print(f"Local training done. Avg loss: {avg_loss:.4f}")
    return model, avg_loss
