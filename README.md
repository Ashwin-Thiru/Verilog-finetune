# Verilog-Finetune

An end-to-end machine learning pipeline for fine-tuning Large Language Models (specifically LLaMA 3.1) for hardware design, Verilog generation, and RTL-to-DFT (Design for Testability) debugging.

This project focuses on processing raw hardware datasets, generating synthetic Verilog code, verifying outputs using automated testbenches, and fine-tuning an LLM using Unsloth for highly efficient hardware generation.

## 📁 Project Structure

*   `scripts/`: Contains the core Python utilities for dataset curation and pipeline management.
    *   **Data Collection:** `download_hf_datasets.py`, `download_chisel.py`, `download_mg.py`.
    *   **Data Processing:** `filter_datasets.py`, `merge_datasets.py`, `downsample_foundation.py`.
    *   **Debugging & Generation:** `generate_debug_dataset.py`, `create_debug_source.py`.
    *   **Testbenches:** `extract_modules_for_tb.py`, `parse_testbenches.py`.
*   `training/`: Contains the core LLM fine-tuning scripts.
    *   `train.py`: The main training script using Unsloth to fine-tune the LLaMA 3.1 4-bit model.
*   `testbench_source/`: Holds the source Verilog testbenches used to validate synthetic module correctness and extract error feedback for the training loop.
*   `testbench_responses/` *(git-ignored)*: Execution logs and outputs from running testbenches.
*   `raw_datasets/` & `filtered/` *(git-ignored)*: Massive raw dataset files (e.g., `.arrow` tables).
*   `checkpoints/` & `verilog-agent-llama31-4bit-final/` *(git-ignored)*: The intermediate weights and the final exported model artifacts.

## 🚀 Pipeline Workflow

1.  **Dataset Acquisition:** Raw datasets are pulled from Hugging Face and other hardware repositories (Chisel, MG) using the `scripts/download_*.py` tools.
2.  **Dataset Refinement:** We run deep filtering (`filter_datasets.py`, `filter_foundation.py`) and data merging (`merge_datasets.py`) to create a high-quality, sanitized Verilog knowledge base.
3.  **Testbench-Driven Debugging:** `generate_debug_dataset.py` creates a dataset of buggy Verilog alongside its compiler/simulation feedback by using automated testbenches.
4.  **Model Fine-Tuning:** The processed and tokenized `final_training.jsonl` data is fed into `training/train.py`, which leverages Unsloth for ultra-fast, memory-efficient 4-bit LoRA fine-tuning of LLaMA 3.1.
5.  **Export:** The fine-tuned agent is exported for integration with the main `verilog-agent` inference pipeline.

## 🛑 Important Note on Datasets & Weights
Due to file size constraints, raw datasets (`.arrow`), intermediate training checkpoints, and the final model weights are intentionally ignored from version control via `.gitignore`. 

*Model weights and final datasets are hosted externally (e.g., on HuggingFace).*

## ⚙️ Setup and Usage

**Requirements:**
*   Python 3.10+
*   Unsloth
*   PyTorch
*   Hugging Face `datasets` and `transformers`

To start a training run locally:
```bash
cd training/
python train.py
