import glob, torch

paths = sorted(glob.glob("data/client_*/data.pt")) + ["data/validation/data.pt"]
for p in paths:
    ds = torch.load(p) # works on host where cache exists
    out = []
    for x in ds:
        out.append({
        "input_ids": x["input_ids"].clone() if hasattr(x["input_ids"], "clone") else x["input_ids"],
        "attention_mask": x["attention_mask"].clone() if hasattr(x["attention_mask"], "clone") else x["attention_mask"],
        "label": x["label"].clone() if hasattr(x["label"], "clone") else x["label"],
        })
    torch.save(out, p)
    print("rewrote", p, "examples:", len(out))
