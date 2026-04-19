import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dp_utils import make_private, get_epsilon
 
def local_train(model, dataset, config):
    device = torch.device("cpu")
    model.to(device)
    model.train()
 
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=0,
    )
 
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 2e-4),
    )
 
    # Wrap with Opacus if DP is enabled
    model, optimizer, loader, privacy_engine = make_private(
        model, optimizer, loader, config
    )
 
    total_loss, num_steps = 0.0, 0
 
    for epoch in range(config.get("local_epochs", 2)):
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
 
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
 
            # Opacus handles gradient clipping internally
            # DO NOT call clip_grad_norm_ manually when DP is active
            optimizer.step()
            optimizer.zero_grad()
 
            total_loss += loss.item()
            num_steps += 1
 
    avg_loss = total_loss / num_steps
    epsilon  = get_epsilon(privacy_engine, delta=config.get("target_delta", 1e-5))
 
    print(f"Train done. Loss={avg_loss:.4f}, epsilon={epsilon:.3f}")
    return model, avg_loss, epsilon
