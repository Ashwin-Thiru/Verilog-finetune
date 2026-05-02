# Verilog-Finetune

An end-to-end machine learning pipeline for fine-tuning Large Language Models (specifically **LLaMA 3.1 8B**) for hardware design, Verilog generation, and RTL-to-DFT (Design for Testability) debugging.

This project covers the full lifecycle: raw dataset acquisition, synthetic Verilog generation, automated testbench verification, QLoRA fine-tuning on Kaggle's free T4 GPU, and publishing the final model to HuggingFace Hub in both LoRA adapter and GGUF formats.

---

## 📁 Project Structure

```
Verilog-Finetune/
├── scripts/                        # Core Python utilities for dataset curation
│   ├── download_hf_datasets.py     # Pull datasets from HuggingFace Hub
│   ├── download_chisel.py          # Download Chisel hardware datasets
│   ├── download_mg.py              # Download MG hardware datasets
│   ├── filter_datasets.py          # Deep quality filtering of raw data
│   ├── filter_foundation.py        # Foundation model dataset filtering
│   ├── merge_datasets.py           # Merge multiple sources into one corpus
│   ├── downsample_foundation.py    # Downsample large foundation datasets
│   ├── generate_debug_dataset.py   # Generate buggy Verilog + compiler feedback pairs
│   ├── create_debug_source.py      # Build debug source data from raw Verilog
│   ├── extract_modules_for_tb.py   # Extract Verilog modules for testbench use
│   └── parse_testbenches.py        # Parse and validate testbench outputs
│
├── training/
│   └── train.py                    # Main fine-tuning script (Unsloth + QLoRA)
│
├── testbench_source/               # Source Verilog testbenches for validation
├── testbench_responses/            # Testbench execution logs (git-ignored)
├── raw_datasets/                   # Raw .arrow dataset files (git-ignored)
├── filtered/                       # Filtered dataset outputs (git-ignored)
├── checkpoints/                    # Intermediate training checkpoints (git-ignored)
└── verilog-agent-llama31-4bit-final/  # Final exported model artifacts (git-ignored)
```

---

## 🚀 Pipeline Workflow

```
Raw Hardware Data (HuggingFace, Chisel, MG)
            ↓
    Dataset Acquisition
    (download_*.py scripts)
            ↓
    Dataset Refinement
    (filter, merge, downsample)
            ↓
    Testbench-Driven Debug Generation
    (buggy Verilog + compiler feedback pairs)
            ↓
    Format to Instruction/Chat JSONL
    (system / user / assistant messages)
            ↓
    Token-length Filtering (max 1024 tokens)
            ↓
    QLoRA Fine-Tuning on Kaggle T4 GPU
    (Unsloth + SFTTrainer + LLaMA 3.1 8B 4-bit)
            ↓
    ┌──────────────────────────────┐
    │  LoRA Adapter  │  GGUF q4_k_m│
    └──────────────────────────────┘
            ↓
    Published to HuggingFace Hub
```

### Step-by-step

1. **Dataset Acquisition** — Raw datasets are pulled from HuggingFace and hardware repositories (Chisel, MG) using the `scripts/download_*.py` tools.

2. **Dataset Refinement** — Deep filtering (`filter_datasets.py`, `filter_foundation.py`) and merging (`merge_datasets.py`) produce a high-quality, sanitized Verilog knowledge base.

3. **Testbench-Driven Debugging** — `generate_debug_dataset.py` creates a dataset of buggy Verilog alongside its compiler/simulation feedback using automated testbenches, teaching the model to understand and fix RTL errors.

4. **Dataset Formatting** — All data is formatted into instruction/chat format (system, user, assistant message triples) and serialized to `final_training.jsonl` and `final_validation.jsonl`. Examples exceeding 1024 tokens are filtered out.

5. **Model Fine-Tuning** — `training/train.py` uses Unsloth for ultra-fast, memory-efficient 4-bit QLoRA fine-tuning of LLaMA 3.1 8B on Kaggle's free T4 GPU (16GB VRAM). Only ~0.1% of model parameters are trained via LoRA adapters injected into 7 projection modules across all 32 transformer layers.

6. **Export** — The fine-tuned model is saved as a LoRA adapter and exported to GGUF (q4_k_m quantization) for local inference via Ollama or LM Studio.

7. **Publishing** — The adapter, GGUF model, and dataset are all pushed to HuggingFace Hub.

---

## 🧠 Model & Fine-Tuning Details

| Property | Value |
|---|---|
| Base Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Quantization | 4-bit (QLoRA via BitsAndBytes) |
| Fine-Tuning Method | LoRA — Parameter Efficient Fine-Tuning |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max Sequence Length | 1024 tokens |
| Effective Batch Size | 16 (batch=1, gradient accumulation=16) |
| Training Epochs | 3 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine with 5% warmup |
| Optimizer | AdamW 8-bit |
| Precision | fp16 (auto-detected for T4) |
| Training Hardware | Kaggle — NVIDIA T4 GPU (16GB VRAM) |
| Framework | Unsloth + HuggingFace TRL + PEFT |

### Why QLoRA?

Full fine-tuning of an 8B model requires 80GB+ VRAM. QLoRA makes this feasible on a single free 16GB GPU by:
- Loading the base model in 4-bit quantization (~5GB VRAM instead of ~16GB)
- Training only small LoRA adapter matrices (~0.1% of total parameters)
- Using gradient checkpointing and 8-bit AdamW to further reduce memory usage

---

## 📊 Dataset

The training data combines multiple sources processed into a unified instruction/chat format:

```jsonc
{
  "messages": [
    { "role": "system",    "content": "You are a Verilog expert assistant..." },
    { "role": "user",      "content": "Write a Verilog module for a 4-bit counter" },
    { "role": "assistant", "content": "module counter_4bit ..." }
  ]
}
```

**Dataset sources:**
- HuggingFace hardware/Verilog datasets
- Chisel hardware description repositories
- MG hardware repositories
- Synthetic debug pairs (buggy Verilog + compiler feedback, generated via testbenches)

**Published dataset:** [HuggingFace — your-username/verilog-llama-3.1-dataset](https://huggingface.co/datasets/your-username/verilog-llama-3.1-dataset)

---

## 🤗 HuggingFace Hub

All model artifacts are publicly available:

| Resource | Link |
|---|---|
| GGUF Model (q4_k_m) | [your-username/verilog-llama-3.1-gguf](https://huggingface.co/TheMightyMaddy/verilog-agent-llama31-q4/tree/main) |

---

## 🖥️ Running the Model

### Option 1 — Local inference with Ollama (easiest)

```bash
ollama run hf.co/TheMightyMaddy/verilog-llama-3.1-gguf
```

Works on both GPU and CPU. Ollama automatically uses your GPU if available.

### Option 2 — Google Colab

```python
!pip install unsloth peft --upgrade --quiet

from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit   = True,
)

# Load fine-tuned adapter from HuggingFace
model = PeftModel.from_pretrained(model, "your-username/verilog-llama-3.1-adapter")
FastLanguageModel.for_inference(model)

# Run inference
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Verilog module for a 4-bit counter"}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 3 — Local Python (adapter)

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "your-username/verilog-llama-3.1-adapter")
FastLanguageModel.for_inference(model)
```

---

## ⚙️ Setup and Training

**Requirements:**
- Python 3.10+
- NVIDIA GPU (T4 or better recommended)
- Kaggle account (for free T4 GPU access — 30 hrs/week)

**Install dependencies:**

```bash
pip install unsloth unsloth_zoo hf_transfer triton tyro
# HuggingFace stack (pre-installed on Kaggle)
pip install transformers trl peft accelerate bitsandbytes datasets
```

**Run training on Kaggle:**

1. Upload your dataset to a Kaggle dataset
2. Add your HuggingFace token to Kaggle Secrets as `HF_TOKEN`
3. Open a Kaggle notebook with T4 GPU accelerator enabled
4. Run `training/train.py`

```bash
cd training/
python train.py
```

**Kaggle tips:**
- Keep your browser tab open during training to prevent idle disconnection
- T4 is preferred over P100 — better fp16 tensor cores and Unsloth CUDA kernel compatibility
- Maximum session length is 12 hours; 30 GPU hours available per week on free tier

---

## 🛑 Important Note on Datasets & Weights

Due to file size constraints, the following are excluded from version control via `.gitignore`:
- `raw_datasets/` and `filtered/` — raw `.arrow` table files
- `testbench_responses/` — testbench execution logs
- `checkpoints/` — intermediate training checkpoints
- `verilog-agent-llama31-4bit-final/` — final model weights

**Model weights and final datasets are hosted on HuggingFace Hub** (links above).

---

## 🔧 Output Formats

| Format | Size | Use Case |
|---|---|---|
| LoRA Adapter | ~100MB | Load on top of base model for inference or further training |
| GGUF q4_k_m | ~4-5GB | Standalone local inference via Ollama, LM Studio, llama.cpp |

The GGUF export merges the LoRA adapter back into the base model before quantization, producing a single standalone file — no separate adapter needed at inference time.
