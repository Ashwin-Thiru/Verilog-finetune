# scripts/parse_testbenches.py
import json
import os
import re
import random

random.seed(42)

RESPONSES_DIR = "testbench_responses"
SOURCE_FILE   = "filtered/foundation_3k.jsonl"
OUTPUT_FILE   = "synthetic/testbench_dataset.jsonl"

os.makedirs("synthetic", exist_ok=True)

# ── System Prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Verilog verification engineer.

Rules:
- INSTANTIATE the module under test correctly with all ports
- GENERATE clock and reset sequences
- APPLY diverse test vectors including edge cases
- WRITE self-checking assertions with $display PASS/FAIL
- Use Verilog-2001 ONLY — no SystemVerilog constructs
- End simulation with $finish

OUTPUT FORMAT:
### Testbench Strategy
[what cases are being tested]

### Testbench Code
```verilog
[complete testbench]
```

### Test Case Summary
[list of test cases covered]"""

# ── Load Source Modules (to pair with testbenches) ─────────────

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]

source_modules = load_jsonl(SOURCE_FILE)
print(f"Source modules loaded: {len(source_modules)}")

# ── Parse Response Files ────────────────────────────────────────

def extract_testbenches_from_file(filepath):
    """Extract individual testbenches from one response file."""
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split on === TESTBENCH X === markers
    # Handle variations like === TESTBENCH 1 ===, ===TESTBENCH 1===
    parts = re.split(r'===\s*TESTBENCH\s*\d+\s*===', content)

    testbenches = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Must contain a module definition
        if 'module tb_' not in part and 'module TB_' not in part:
            # Try any testbench module name
            if 'initial begin' not in part:
                continue

        # Clean up — remove leading `timescale if split left it
        # attached to previous block
        code = part.strip()

        # Add timescale if missing
        if '`timescale' not in code:
            code = '`timescale 1ns/1ps\n' + code

        # Must have basic testbench elements
        if 'initial begin' not in code:
            continue
        if '$finish' not in code and '$stop' not in code:
            continue

        testbenches.append(code)

    return testbenches

# ── Extract Module Name From Testbench ─────────────────────────

def extract_dut_name(tb_code):
    """Extract the DUT module name from instantiation."""
    # Look for: module_name dut (
    match = re.search(r'(\w+)\s+dut\s*\(', tb_code)
    if match:
        return match.group(1)
    # Fallback: look for tb_ prefix and strip it
    match = re.search(r'module\s+tb_(\w+)', tb_code)
    if match:
        return match.group(1)
    return None

# ── Generate Training Example ───────────────────────────────────

def make_testbench_example(dut_module_code, tb_code,
                            dut_name, topic):
    # Extract module name from DUT for strategy text
    mod_match = re.search(r'module\s+(\w+)', dut_module_code)
    mod_name = mod_match.group(1) if mod_match else dut_name

    # Count test cases in testbench
    pass_count = tb_code.count('$display("PASS')
    fail_count = tb_code.count('$display("FAIL')

    strategy = (
        f"Testing {mod_name} module for correct "
        f"reset behavior, normal operation, and edge cases. "
        f"The testbench includes {pass_count} self-checking "
        f"assertions with PASS/FAIL reporting."
    )

    summary_lines = []
    # Extract test comments as summary
    test_comments = re.findall(
        r'//\s*(Test\s*\d+[^:\n]*:?[^\n]*)', tb_code
    )
    for i, comment in enumerate(test_comments[:6]):
        summary_lines.append(f"- {comment.strip()}")
    if not summary_lines:
        summary_lines = [
            "- Reset behavior verification",
            "- Normal operation testing",
            "- Edge case coverage"
        ]

    assistant_msg = (
        f"### Testbench Strategy\n"
        f"{strategy}\n\n"
        f"### Testbench Code\n"
        f"```verilog\n{tb_code}\n```\n\n"
        f"### Test Case Summary\n"
        + '\n'.join(summary_lines)
    )

    user_msg = (
        f"## Module to Verify\n"
        f"```verilog\n{dut_module_code}\n```\n\n"
        f"## Instructions\n"
        f"Write a comprehensive self-checking testbench "
        f"for this Verilog module. Include reset behavior, "
        f"normal operation, and edge cases. Use $display "
        f"for PASS/FAIL reporting."
    )

    return {
        "task": "testbench",
        "topic": topic,
        "dut_name": dut_name or "unknown",
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

# ── Main ────────────────────────────────────────────────────────

def main():
    # Collect all testbenches from response files
    all_testbenches = []
    response_files = sorted([
        f for f in os.listdir(RESPONSES_DIR)
        if f.endswith('_response.txt')
    ])

    print(f"\nParsing {len(response_files)} response files...")
    for fname in response_files:
        fpath = os.path.join(RESPONSES_DIR, fname)
        tbs = extract_testbenches_from_file(fpath)
        all_testbenches.extend(tbs)
        print(f"  {fname:35s}  {len(tbs)} testbenches")

    print(f"\nTotal testbenches extracted: {len(all_testbenches)}")

    # Match testbenches with source modules
    # Strategy: match by DUT name against module names in source
    # Build lookup of source modules by module name
    source_by_name = {}
    for item in source_modules:
        code = item.get('code', '')
        match = re.search(r'module\s+(\w+)', code)
        if match:
            source_by_name[match.group(1)] = item

    print(f"Source modules indexed: {len(source_by_name)}")

    # Try to match testbenches to source modules
    matched   = []
    unmatched = []

    for tb_code in all_testbenches:
        dut_name = extract_dut_name(tb_code)
        if dut_name and dut_name in source_by_name:
            source = source_by_name[dut_name]
            matched.append((tb_code, source))
        else:
            unmatched.append(tb_code)

    print(f"Matched to source:   {len(matched)}")
    print(f"Unmatched:           {len(unmatched)}")

    # For unmatched — pair with random source module
    # (still valid training data — teaches format)
    random_sources = random.choices(source_modules,
                                    k=len(unmatched))
    for tb_code, source in zip(unmatched, random_sources):
        matched.append((tb_code, source))

    print(f"Total pairs:         {len(matched)}")

    # Generate training examples
    examples = []
    for tb_code, source in matched:
        dut_name = extract_dut_name(tb_code)
        topic    = source.get('topic', 'other')
        dut_code = source.get('code', '')

        ex = make_testbench_example(
            dut_code, tb_code, dut_name, topic
        )
        examples.append(ex)

    # Shuffle and save
    random.shuffle(examples)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved {len(examples)} testbench examples")
    print(f"Output: {OUTPUT_FILE}")

    # Topic breakdown
    from collections import Counter
    topics = Counter(ex['topic'] for ex in examples)
    print("\nTopic breakdown:")
    for topic, count in sorted(topics.items(),
                                key=lambda x: -x[1]):
        print(f"  {topic:20s}  {count}")

if __name__ == "__main__":
    main()