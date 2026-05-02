# scripts/peek_datasets.py
from datasets import load_from_disk

# Peek Chisel
print("═══ CHISEL ═══")
chisel = load_from_disk("raw_datasets/chisel_verilog")
print("Splits:", chisel.keys() if hasattr(chisel, 'keys') else "no splits")

# Handle both cases — with splits and without
if hasattr(chisel, 'keys'):
    data = chisel[list(chisel.keys())[0]]  # take first split
else:
    data = chisel

print("Columns:", data.column_names)
print("First entry:\n", data[0])
print()

# Peek MG-Verilog
print("═══ MG-VERILOG ═══")
mg = load_from_disk("raw_datasets/mg_verilog")
if hasattr(mg, 'keys'):
    data_mg = mg[list(mg.keys())[0]]
else:
    data_mg = mg

print("Columns:", data_mg.column_names)
print("First entry:\n", data_mg[0])