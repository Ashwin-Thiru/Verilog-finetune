import json
import random

DEBUG_PATH      = "/home/ashwin/verilog-finetune/synthetic/debug_dataset_3k.jsonl"
TESTBENCH_PATH  = "/home/ashwin/verilog-finetune/synthetic/testbench_dataset.jsonl"
FOUNDATION_PATH = "/home/ashwin/verilog-finetune/filtered/foundation_converted.jsonl"

TRAIN_OUTPUT    = "/home/ashwin/verilog-finetune/final_training.jsonl"
VAL_OUTPUT      = "/home/ashwin/verilog-finetune/final_validation.jsonl"

SEED            = 42
SPLIT_RATIO     = 0.9  # 90% train, 10% val

# ─────────────────────────────────────────────
def load_jsonl(path, label):
    entries = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Validate must have messages[]
                msgs = entry.get("messages", [])
                roles = [m.get("role") for m in msgs]
                if "system" in roles and "user" in roles and "assistant" in roles:
                    entries.append(entry)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    print(f"  {label:<30} : {len(entries):>5} loaded  |  {skipped} skipped")
    return entries

# ─────────────────────────────────────────────
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# ─────────────────────────────────────────────
def main():
    random.seed(SEED)

    print("=" * 55)
    print("  Verilog Fine-tune Dataset Merge & Split")
    print("=" * 55)

    # ── Load ──────────────────────────────────
    print("\n[1/3] Loading datasets...")
    debug_data      = load_jsonl(DEBUG_PATH,      "debug_dataset_3k")
    testbench_data  = load_jsonl(TESTBENCH_PATH,  "testbench_dataset")
    foundation_data = load_jsonl(FOUNDATION_PATH, "foundation_converted")

    all_data = debug_data + testbench_data + foundation_data
    print(f"\n  Total combined   : {len(all_data)}")

    # ── Shuffle & Split ───────────────────────
    print("\n[2/3] Shuffling and splitting...")
    random.shuffle(all_data)

    split_idx  = int(len(all_data) * SPLIT_RATIO)
    train_data = all_data[:split_idx]
    val_data   = all_data[split_idx:]

    print(f"  Train ({int(SPLIT_RATIO*100)}%)  : {len(train_data)}")
    print(f"  Val   ({int((1-SPLIT_RATIO)*100)}%)  : {len(val_data)}")

    # ── Task distribution ─────────────────────
    print("\n  Task distribution in train set:")
    task_counts = {}
    for e in train_data:
        t = e.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    for task, count in sorted(task_counts.items()):
        pct = count / len(train_data) * 100
        print(f"    {task:<15} : {count:>5}  ({pct:.1f}%)")

    print("\n  Task distribution in val set:")
    task_counts_val = {}
    for e in val_data:
        t = e.get("task", "unknown")
        task_counts_val[t] = task_counts_val.get(t, 0) + 1
    for task, count in sorted(task_counts_val.items()):
        pct = count / len(val_data) * 100
        print(f"    {task:<15} : {count:>5}  ({pct:.1f}%)")

    # ── Save ──────────────────────────────────
    print("\n[3/3] Saving...")
    save_jsonl(train_data, TRAIN_OUTPUT)
    save_jsonl(val_data,   VAL_OUTPUT)
    print(f"  Saved: final_training.jsonl    ({len(train_data)} entries)")
    print(f"  Saved: final_validation.jsonl  ({len(val_data)} entries)")

    print("\n" + "=" * 55)
    print("  Done! Ready for Unsloth fine-tuning.")
    print("=" * 55)

if __name__ == "__main__":
    main()