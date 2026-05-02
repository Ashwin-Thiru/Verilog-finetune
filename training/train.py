import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
import torch
import gc
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 768

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = [
        "q_proj", "v_proj",    # minimal — attention only
    ],
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Aggressive memory cleanup
torch.cuda.empty_cache()
gc.collect()

dataset = load_dataset(
    "json",
    data_files = "/home/ashwin/verilog-finetune/training/final_training.jsonl",
    split = "train"
)
print(f"Dataset loaded: {len(dataset)} examples")

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def formatting_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=False
        )
        for msg in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(
    formatting_func,
    batched = True,
    remove_columns = dataset.column_names
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 64,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        output_dir = "verilog-agent-llama31-4bit-final",
        report_to = "none",
        save_strategy = "epoch",
        save_total_limit = 2,
    ),
)

print("Starting fine-tuning...")
trainer_stats = trainer.train()

model.save_pretrained("verilog-agent-llama31-4bit-final")
tokenizer.save_pretrained("verilog-agent-llama31-4bit-final")

print(f"Training complete!")
print(f"Peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
print("Model saved to verilog-agent-llama31-4bit-final/")