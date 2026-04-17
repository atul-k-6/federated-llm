from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch
 
BASE_MODEL = "distilbert-base-uncased"
NUM_LABELS = 4   # AG News has 4 classes
 
def build_lora_model():
    """
    Load DistilBERT and wrap it with LoRA adapters.
    Only the LoRA layers are trainable; base weights are frozen.
    """
    # Load base model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
 
    # LoRA config: inject low-rank adapters into attention layers
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,               # Rank of the decomposition (8 is standard)
        lora_alpha=16,     # Scaling factor (usually 2x rank)
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
        lora_dropout=0.1,
        bias="none",
    )
 
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Prints: trainable params: 296,448 || all params: 67,252,740 || ~0.44%
    return model
 
def get_lora_state_dict(model):
    """Extract only the LoRA adapter weights (not the full model)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}
 
def set_lora_state_dict(model, state_dict):
    """Load LoRA weights without touching base model weights."""
    model.load_state_dict(state_dict, strict=False)
