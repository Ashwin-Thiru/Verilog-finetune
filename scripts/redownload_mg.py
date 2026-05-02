# scripts/redownload_mg.py  — replace entire file
import pyarrow as pa
import pyarrow.ipc as ipc
import os

arrow_path = "raw_datasets/mg_verilog/train/data-00000-of-00001.arrow"

print(f"File size: {os.path.getsize(arrow_path)} bytes")
print()

# Read arrow file directly
with open(arrow_path, 'rb') as f:
    reader = ipc.open_file(f)
    print(f"Number of record batches: {reader.num_record_batches}")
    
    table = reader.read_all()
    print(f"Schema:\n{table.schema}")
    print(f"Total rows: {len(table)}")
    
    if len(table) > 0:
        print("\nFirst row:")
        first = {col: table[col][0].as_py() for col in table.schema.names}
        for k, v in first.items():
            print(f"  {k}: {str(v)[:200]}")