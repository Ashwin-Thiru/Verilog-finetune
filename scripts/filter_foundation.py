import json

INPUT_PATH  = "/home/ashwin/verilog-finetune/filtered/foundation_3k.jsonl"
OUTPUT_PATH = "/home/ashwin/verilog-finetune/filtered/foundation_clean.jsonl"

print("=" * 50)
print("  Foundation Dataset Filter")
print("=" * 50)

total      = 0
kept       = 0
dropped    = 0
source_counts = {}

with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        line = line.strip()
        if not line:
            continue

        total += 1

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [SKIP] Line {i+1}: JSON error - {e}")
            dropped += 1
            continue

        source = entry.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

        # Drop verilogeval entirely
        if source == "verilogeval":
            dropped += 1
            continue

        fout.write(json.dumps(entry) + "\n")
        kept += 1

print(f"\n  Source breakdown (before filtering):")
for src, count in sorted(source_counts.items()):
    status = "❌ dropped" if src == "verilogeval" else "✅ kept"
    print(f"    {src:<20} : {count:>5}  {status}")

print(f"\n  Total entries  : {total}")
print(f"  Kept           : {kept}")
print(f"  Dropped        : {dropped}")
print(f"\n  Output saved to:")
print(f"  {OUTPUT_PATH}")
print("=" * 50)