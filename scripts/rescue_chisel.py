import json
import os
from tqdm import tqdm
from datasets import Dataset
from collections import Counter
import subprocess
import tempfile

# Configuration
INPUT_DIR = "raw_datasets/chisel_verilog/train"
OUT_PATH = "filtered/chisel_filtered.jsonl"

SV_KEYWORDS = ['always_ff', 'always_comb', 'logic ', 'unique case', 'assert property']

def is_verilog_2001(code):
    return not any(kw in code for kw in SV_KEYWORDS)

def passes_iverilog(code):
    with tempfile.NamedTemporaryFile(suffix='.v', mode='w', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        # Strict 2001 check
        res = subprocess.run(["iverilog", "-g2001", "-o", "/dev/null", fname], 
                             capture_output=True, timeout=5)
        return res.returncode == 0
    except:
        return False
    finally:
        if os.path.exists(fname): os.unlink(fname)

def main():
    print("🚀 Starting Chisel Rescue...")
    
    # Get all shard paths
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.arrow')]
    
    passed_count = 0
    fail_reasons = Counter()

    # Open in 'w' first to clear, then 'a' inside the loop
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        pass

    for shard in files:
        print(f"📦 Processing Shard: {os.path.basename(shard)}")
        try:
            # Load single shard
            ds = Dataset.from_file(shard)
            
            with open(OUT_PATH, 'a', encoding='utf-8') as f:
                for row in tqdm(ds, desc="   Filtering", leave=False):
                    instruction = row.get('prompt', '')
                    code = row.get('response', '') or ''
                    
                    if 'module ' not in code:
                        fail_reasons['no_module'] += 1
                        continue
                    
                    if not is_verilog_2001(code):
                        fail_reasons['systemverilog'] += 1
                        continue
                        
                    if passes_iverilog(code):
                        entry = {
                            "source": "chisel_verilog",
                            "instruction": instruction,
                            "code": code
                        }
                        f.write(json.dumps(entry) + '\n')
                        passed_count += 1
                    else:
                        fail_reasons['syntax_error'] += 1
        except Exception as e:
            print(f"❌ Error reading {shard}: {e}")

    print(f"\n✅ Done! Rescued {passed_count} examples.")
    print(f"Fail Breakdown: {fail_reasons}")

if __name__ == "__main__":
    main()
