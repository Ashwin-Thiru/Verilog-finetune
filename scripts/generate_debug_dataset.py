import json
import os
import re
import random
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

random.seed(42)
os.makedirs("synthetic", exist_ok=True)

# ── System Prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Verilog HDL debugging assistant.

Rules:
- Do NOT change module ports or interface
- Fix ALL errors shown
- Follow Verilog-2001 strictly
- Do NOT use always_ff, always_comb, logic, or any SystemVerilog constructs
- Do NOT repeat a fix strategy that already failed

OUTPUT FORMAT:
### Error Analysis
[explain what went wrong and why]

### Fix Applied
[describe exactly what you changed]

### Corrected Verilog Code
```verilog
[corrected module here]
```

### Confidence
[High/Medium/Low with one line reason]"""

# ── Bug Injectors ───────────────────────────────────────────────

def inject_blocking_nonblocking(code):
    lines = code.split('\n')
    new_lines = []
    inside_seq = False
    changed = False
    for line in lines:
        if re.search(r'always\s*@\s*\(posedge', line):
            inside_seq = True
        if inside_seq and '<=' in line and not changed:
            line = line.replace('<=', '=', 1)
            changed = True
            inside_seq = False
        new_lines.append(line)
    if changed:
        return '\n'.join(new_lines), "blocking_nonblocking"
    return None, None


def inject_undeclared_signal(code):
    if 'endmodule' in code:
        modified = code.replace(
            'endmodule',
            '  assign _undeclared_ghost = 1\'b0;\nendmodule',
            1
        )
        return modified, "undeclared_signal"
    return None, None


def inject_missing_begin_end(code):
    match = re.search(
        r'(always\s*@[^;]+;?\s*\n)(\s*begin\n)',
        code
    )
    if match:
        modified = code.replace(match.group(2), '\n', 1)
        end_idx = modified.find('\n    end',
                                modified.find(match.group(1)))
        if end_idx != -1:
            modified = (modified[:end_idx] +
                       modified[end_idx + 8:])
            return modified, "missing_begin_end"
    return None, None


def inject_wrong_reset_edge(code):
    if 'negedge rst' in code:
        return (
            code.replace('negedge rst', 'posedge rst', 1),
            "reset_edge"
        )
    if 'negedge reset' in code:
        return (
            code.replace('negedge reset', 'posedge reset', 1),
            "reset_edge"
        )
    return None, None


def inject_width_mismatch(code):
    match = re.search(r'output\s+reg\s+\[(\d+):0\]', code)
    if match:
        original = int(match.group(1))
        new_width = original + 2
        modified = code.replace(
            match.group(0),
            f'output reg [{new_width}:0]',
            1
        )
        return modified, "width_mismatch"
    return None, None


def inject_wrong_sensitivity(code):
    match = re.search(r'always\s*@\s*\(([^)]+)\)', code)
    if match:
        sens = match.group(1)
        if 'posedge' not in sens and 'negedge' not in sens:
            signals = [s.strip() for s in sens.split(',')]
            if len(signals) > 1:
                signals.pop()
                new_sens = ', '.join(signals)
                modified = code.replace(
                    match.group(0),
                    f'always @({new_sens})'
                )
                return modified, "sensitivity_list"
    return None, None


INJECTORS = [
    (inject_blocking_nonblocking, 0.25),
    (inject_undeclared_signal,    0.20),
    (inject_missing_begin_end,    0.20),
    (inject_width_mismatch,       0.15),
    (inject_wrong_reset_edge,     0.10),
    (inject_wrong_sensitivity,    0.10),
]

INJECTOR_FUNCTIONS = [f for f, _ in INJECTORS]
INJECTOR_WEIGHTS   = [w for _, w in INJECTORS]

# ── Wrong Fix History Descriptions ─────────────────────────────

WRONG_FIXES = {
    "blocking_nonblocking": [
        "Changed variable declaration from reg to wire",
        "Added extra begin/end block around assignment",
    ],
    "undeclared_signal": [
        "Added input port for the missing signal",
        "Changed assign statement to always block",
    ],
    "missing_begin_end": [
        "Added begin/end in wrong location inside if statement",
        "Wrapped entire always block in extra begin/end",
    ],
    "width_mismatch": [
        "Changed input port width instead of output port",
        "Added sign extension without fixing root cause",
    ],
    "reset_edge": [
        "Changed clock edge instead of reset edge",
        "Inverted reset logic condition inside always block",
    ],
    "sensitivity_list": [
        "Added posedge to sensitivity list instead of level",
        "Converted always block to always @(*) incorrectly",
    ],
}

# ── Verilator Runner ────────────────────────────────────────────

def run_verilator(code):
    with tempfile.NamedTemporaryFile(
        suffix='.v', mode='w',
        delete=False, encoding='utf-8'
    ) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ['verilator', '--lint-only', '-Wall',
             '--language', '1364-2001', fname],
            capture_output=True, text=True, timeout=15
        )
        errors = result.stderr[:2000]
        has_errors = (
            result.returncode != 0 or
            '%Error' in errors or
            '%Warning' in errors
        )
        return errors, has_errors
    except subprocess.TimeoutExpired:
        return "Timeout", False
    except Exception as e:
        return str(e), False
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

# ── Analysis and Fix Templates ──────────────────────────────────

def get_analysis_and_fix(bug_type):
    templates = {
        "blocking_nonblocking": (
            "The Verilator warning indicates a blocking "
            "assignment (`=`) was used inside a sequential "
            "always block (`always @(posedge clk)`). "
            "Blocking assignments in sequential logic cause "
            "race conditions and non-deterministic behavior.",
            "Changed the blocking assignment (`=`) to a "
            "non-blocking assignment (`<=`) inside the "
            "sequential always block."
        ),
        "undeclared_signal": (
            "Verilator reports an undeclared identifier. "
            "The code references a signal that has not been "
            "declared as a wire, reg, or port in this module.",
            "Removed the reference to the undeclared signal. "
            "It was not needed for the module's functionality."
        ),
        "missing_begin_end": (
            "The syntax error indicates missing begin/end "
            "delimiters. When an always block contains more "
            "than one statement, begin/end must wrap them.",
            "Added begin/end delimiters around the always "
            "block body to properly group the statements."
        ),
        "width_mismatch": (
            "Verilator reports a width mismatch. The port "
            "width declaration does not match the width of "
            "the signal being assigned to it.",
            "Corrected the output port width declaration to "
            "match the actual signal width used in the logic."
        ),
        "reset_edge": (
            "The always block sensitivity list uses the wrong "
            "edge for the reset signal. Active-low resets "
            "should use negedge, active-high should use posedge.",
            "Corrected the reset edge in the sensitivity list "
            "to match the reset polarity used in the logic."
        ),
        "sensitivity_list": (
            "The always block sensitivity list is incomplete. "
            "One or more signals read inside the block are "
            "missing, causing incorrect simulation behavior.",
            "Added the missing signals to the always block "
            "sensitivity list to ensure correct simulation."
        ),
    }
    return templates.get(
        bug_type,
        (
            "The Verilator error indicates a syntax or "
            "semantic issue in the Verilog code.",
            "Fixed the identified issue in the module."
        )
    )

# ── Format Into Exact Prompt Template ──────────────────────────

def make_debug_example(broken_code, error_log,
                        fixed_code, attempt,
                        history, topic, bug_type):
    history_text = ""
    if history:
        history_text = (
            "\n## Previous Fix Attempts "
            "(do NOT repeat these)\n"
        )
        for h in history:
            history_text += (
                f"- **Fix attempted:** {h}\n"
                f"  **Result:** failed\n"
            )

    user_msg = (
        f"## Current Verilog Code\n"
        f"```verilog\n{broken_code}\n```\n\n"
        f"## Verilator Errors\n"
        f"{error_log}"
        f"{history_text}\n"
        f"## Instructions\n"
        f"This is fix attempt {attempt} of 3. "
        f"Analyze the errors above, review what was "
        f"already tried, and return a corrected version "
        f"of the Verilog code."
    )

    analysis, fix_desc = get_analysis_and_fix(bug_type)

    assistant_msg = (
        f"### Error Analysis\n"
        f"{analysis}\n\n"
        f"### Fix Applied\n"
        f"{fix_desc}\n\n"
        f"### Corrected Verilog Code\n"
        f"```verilog\n{fixed_code}\n```\n\n"
        f"### Confidence\n"
        f"High. The fix directly addresses "
        f"the Verilator error shown."
    )

    return {
        "task": "debug",
        "attempt": attempt,
        "topic": topic,
        "bug_type": bug_type,
        "messages": [
            {"role": "system",   "content": SYSTEM_PROMPT},
            {"role": "user",     "content": user_msg},
            {"role": "assistant","content": assistant_msg}
        ]
    }

# ── Per-Item Processing ─────────────────────────────────────────

def process_one(item):
    code  = item.get('code', '')
    topic = item.get('topic', 'other')

    # Shuffle injectors for variety
    paired = list(zip(INJECTOR_FUNCTIONS, INJECTOR_WEIGHTS))
    random.shuffle(paired)

    for injector_fn, _ in paired:
        broken, bug_type = injector_fn(code)
        if broken is None:
            continue

        errors, has_errors = run_verilator(broken)
        if not has_errors:
            continue

        wrong_fixes = WRONG_FIXES.get(bug_type, [
            "Applied generic restructuring",
            "Tried alternative signal assignment"
        ])

        results = []

        # Attempt 1 — no history
        results.append(make_debug_example(
            broken, errors, code,
            1, [], topic, bug_type
        ))

        # Attempt 2 — one failed fix
        results.append(make_debug_example(
            broken, errors, code,
            2, [wrong_fixes[0]], topic, bug_type
        ))

        # Attempt 3 — two failed fixes
        results.append(make_debug_example(
            broken, errors, code,
            3, wrong_fixes[:2], topic, bug_type
        ))

        return results  # one bug per module

    return []  # no successful injection

# ── Main ────────────────────────────────────────────────────────

def main():
    with open("filtered/debug_source.jsonl",
              encoding='utf-8') as f:
        source = [json.loads(l) for l in f]

    print(f"Source modules:   {len(source)}")
    print(f"Parallel workers: 4")
    print(f"Expected output:  ~{len(source) * 2} examples\n")

    all_examples   = []
    bug_type_counts = Counter()

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_one, item): item
            for item in source
        }
        for future in tqdm(
            as_completed(futures),
            total=len(source),
            desc="Injecting bugs"
        ):
            try:
                results = future.result()
                all_examples.extend(results)
                for ex in results:
                    bug_type_counts[ex['bug_type']] += 1
            except Exception:
                pass

    print(f"\nTotal examples generated: {len(all_examples)}")
    print(f"\nBug type breakdown:")
    for bug, count in sorted(bug_type_counts.items(),
                              key=lambda x: -x[1]):
        modules = count // 3
        print(f"  {bug:30s}  "
              f"{modules:4d} modules  "
              f"{count:5d} examples")

    random.shuffle(all_examples)

    out_path = "synthetic/debug_dataset.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {out_path}")
    print(f"Total: {len(all_examples)} debug examples")


if __name__ == "__main__":
    main()