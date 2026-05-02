from datasets import load_dataset
import os

# Exact paths inside your project
MG_PATH     = r"D:\verilog-finetune\raw_datasets\mg_verilog"
CHISEL_PATH = r"D:\verilog-finetune\raw_datasets\chisel_verilog"

os.makedirs(MG_PATH, exist_ok=True)
os.makedirs(CHISEL_PATH, exist_ok=True)

print("Downloading MG-Verilog...")
mg = load_dataset("GaTech-EIC/MG-Verilog")
mg.save_to_disk(MG_PATH)
print(f"MG-Verilog saved to {MG_PATH}")
print(f"Splits: {list(mg.keys())}")
print(f"First entry keys: {list(mg['train'][0].keys())}\n")

print("Downloading Chisel-Verilog...")
chisel = load_dataset("rtl-llm/chisel-verilog-pairs")
chisel.save_to_disk(CHISEL_PATH)
print(f"Chisel saved to {CHISEL_PATH}")
print(f"Splits: {list(chisel.keys())}")
print(f"First entry keys: {list(chisel['train'][0].keys())}\n")

print("All done!")