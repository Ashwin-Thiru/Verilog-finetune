import json

INPUT_PATH  = "/home/ashwin/verilog-finetune/filtered/foundation_clean.jsonl"
OUTPUT_PATH = "/home/ashwin/verilog-finetune/filtered/foundation_converted.jsonl"

SYSTEM_PROMPT = """You are an expert Verilog HDL designer.

Rules:
- Write clean, synthesizable Verilog-2001 code
- Do NOT use SystemVerilog constructs (always_ff, always_comb, logic, etc.)
- Follow Verilog-2001 strictly
- Ensure the module is complete and functional
- Do NOT change module ports or interfaces

OUTPUT FORMAT:
### Design Explanation
[brief explanation of the design approach]

### Verilog Code
```verilog
[complete module here]
```"""

print("=" * 50)
print("  Foundation Dataset Conversion")
print("=" * 50)

total    = 0
converted = 0
skipped  = 0

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
            skipped += 1
            continue

        instruction = entry.get("instruction", "").strip()
        code        = entry.get("code", "").strip()
        topic       = entry.get("topic", "general")
        source      = entry.get("source", "unknown")

        # Skip if missing key fields
        if not instruction or not code:
            print(f"  [SKIP] Line {i+1}: missing instruction or code")
            skipped += 1
            continue

        # Wrap code in verilog block if not already wrapped
        if "```verilog" in code:
            assistant_content = code
        else:
            assistant_content = f"```verilog\n{code}\n```"

        converted_entry = {
            "task": "foundation",
            "source": source,
            "topic": topic,
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": assistant_content}
            ]
        }

        fout.write(json.dumps(converted_entry) + "\n")
        converted += 1

print(f"\n  Total entries  : {total}")
print(f"  Converted      : {converted}")
print(f"  Skipped        : {skipped}")
print(f"\n  Output saved to:")
print(f"  {OUTPUT_PATH}")
print("=" * 50)
print("\n  Sample converted entry (first one):")
print("-" * 50)

# Print first entry for verification
with open(OUTPUT_PATH, "r") as f:
    first = json.loads(f.readline())
    print(f"  task   : {first['task']}")
    print(f"  source : {first['source']}")
    print(f"  topic  : {first['topic']}")
    print(f"  roles  : {[m['role'] for m in first['messages']]}")
    print(f"  user   : {first['messages'][1]['content'][:80]}...")
    print(f"  asst   : {first['messages'][2]['content'][:80]}...")