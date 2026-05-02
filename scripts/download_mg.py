from datasets import load_dataset

ds = load_dataset("GaTech-EIC/MG-Verilog")
ds.save_to_disk("../raw_datasets/mg_verilog")
print("Done. Splits:", ds.keys())
print("Sample entry keys:", ds['train'][0].keys())