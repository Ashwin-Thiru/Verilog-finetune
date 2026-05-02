# scripts/peek_mg_fix.py  — replace entire file with this
from datasets import Dataset
import os

print("═══ MG-VERILOG PROPER LOAD ═══")

arrow_path = "raw_datasets/mg_verilog/train/data-00000-of-00001.arrow"

ds = Dataset.from_file(arrow_path)
print("Columns:", ds.column_names)
print("Size:", len(ds))
print()
print("First entry:")
for key, val in ds[0].items():
    # truncate long values so it's readable
    val_str = str(val)[:200]
    print(f"  {key}: {val_str}")