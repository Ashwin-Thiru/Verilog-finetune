from datasets import load_dataset

ds = load_dataset("rtl-llm/chisel-verilog-pairs")
ds.save_to_disk("../raw_datasets/chisel_verilog")
print("Done.")
print("Sample:", ds['train'][0])