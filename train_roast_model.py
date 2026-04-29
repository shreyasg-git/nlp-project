from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Configuration for RTX 2070 Super (8GB)
max_seq_length = 2048 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage to ~7.5GB

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (The Fine-Tuning Layer)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: higher = more parameters to train, more VRAM used
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized to 0 for Unsloth
    bias = "none",    # Optimized to "none" for Unsloth
    use_gradient_checkpointing = "unsloth", # Use 70% less VRAM
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Load & Prepare Dataset
dataset = load_dataset("json", data_files={"train": "llama3-roast-dataset.jsonl"}, split="train")

# 5. Initialize Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # This script assumes data is pre-formatted, or use a formatting function
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Total training steps. Set to None and use num_train_epochs for full passes
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Train
print("Starting training on RTX 2070 Super...")
trainer_stats = trainer.train()

# 7. Save Model
model.save_pretrained("llama3_roast_model") # Local saving
tokenizer.save_pretrained("llama3_roast_model")
print("Training complete. Model saved to 'llama3_roast_model'.")
