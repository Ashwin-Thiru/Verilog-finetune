import json
import subprocess
import tempfile
import os
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────

os.makedirs("filtered", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

IVERILOG_WORKERS = 2   # Only 2 parallel iverilog processes — safe for WSL
CHUNK_SIZE = 50        # 50 at a time — much lower RAM spike

SV_KEYWORDS = [
    'always_ff', 'always_comb', 'always_latch',
    'logic ', 'bit ', 'byte ', 'shortint',
    'typedef', 'enum ', 'struct ',
    'interface', 'modport', 'clocking',
    'unique case', 'priority case',
    'import ', '::',
]

# ── Checkers ───────────────────────────────────────────────────

def is_verilog_2001(code):
    for kw in SV_KEYWORDS:
        if kw in code:
            return False
    return True

def passes_iverilog(code):
    with tempfile.NamedTemporaryFile(
        suffix='.v', mode='w', delete=False, encoding='utf-8'
    ) as f:
        f.write(code)
        fname = f.name
    try:
        # ulimit -v 524288 = 512MB virtual memory cap per ivl process.
        # preexec_fn / resource.setrlimit do NOT work on WSL2 — the kernel
        # ignores RLIMIT_AS for child processes. Shell ulimit is the only
        # reliable enforcement mechanism on WSL2.
        cmd = f"ulimit -v 524288 && ulimit -t 10 && iverilog -o /dev/null -g2001 {fname}"
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=15
        )
        ok = result.returncode == 0
        return ok, "ok" if ok else "syntax_error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        print("  WARNING: iverilog not found — skipping syntax check")
        return True, "no_iverilog"
    except Exception as e:
        return False, f"error:{str(e)[:40]}"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass

def has_module(code):
    return 'module ' in code and 'endmodule' in code

# ── Checkpoint helpers ─────────────────────────────────────────

def load_checkpoint(name):
    path = f"checkpoints/{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(name, done_keys):
    path = f"checkpoints/{name}.json"
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(list(done_keys), f)
    os.replace(tmp, path)

# ── Chunk filter ───────────────────────────────────────────────

def filter_chunk(chunk):
    passed = []
    fail_reasons = []
    with ThreadPoolExecutor(max_workers=IVERILOG_WORKERS) as pool:
        futures = {pool.submit(passes_iverilog, item["code"]): item for item in chunk}
        for future in as_completed(futures):
            item = futures[future]
            try:
                ok, reason = future.result()
            except Exception as e:
                ok, reason = False, f"future_error:{str(e)[:40]}"
            if ok:
                passed.append(item)
            else:
                fail_reasons.append(reason)
    return passed, fail_reasons

# ── Processors ─────────────────────────────────────────────────

def process_rtlcoder():
    print("\n── RTLCoder ──────────────────────────────")

    path = "raw_datasets/rtlcoder/dataset/Resyn27k.json"
    out_path = "filtered/rtlcoder_filtered.jsonl"

    with open(path, encoding='utf-8') as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        data = []
        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping line {lineno}: {e}")

    del raw
    gc.collect()

    print(f"Total entries: {len(data)}")

    done = load_checkpoint("rtlcoder")
    fail_reasons = []
    candidates = []

    for idx, item in enumerate(tqdm(data, desc="  pre-filter")):
        instruction = item.get("Instruction", "")
        response = item.get("Response", [])
        code = response[0] if isinstance(response, list) and response else str(response)

        if not has_module(code):
            fail_reasons.append("no_module")
            continue
        if not is_verilog_2001(code):
            fail_reasons.append("systemverilog")
            continue

        candidates.append({
            "_idx": idx,
            "source": "rtlcoder",
            "instruction": instruction,
            "code": code
        })

    del data
    gc.collect()

    print(f"  Pre-filter passed: {len(candidates)} candidates → running iverilog...")

    new_done = set(done)
    total_passed = 0
    all_fail_reasons = list(fail_reasons)

    with open(out_path, 'a', encoding='utf-8') as out_file:
        for i in range(0, len(candidates), CHUNK_SIZE):
            chunk_key = f"chunk_{i}"
            if chunk_key in new_done:
                print(f"  skipping chunk {i // CHUNK_SIZE + 1} (already done)")
                continue

            chunk = candidates[i:i + CHUNK_SIZE]
            passed, chunk_fail_reasons = filter_chunk(chunk)
            all_fail_reasons.extend(chunk_fail_reasons)
            total_passed += len(passed)

            for ex in passed:
                ex.pop("_idx", None)
                out_file.write(json.dumps(ex) + '\n')
            out_file.flush()

            new_done.add(chunk_key)
            save_checkpoint("rtlcoder", new_done)
            print(f"  chunk {i // CHUNK_SIZE + 1}: {len(passed)}/{len(chunk)} passed ✓ saved")
            gc.collect()

    print(f"  Total passed: {total_passed}")
    print(f"  Fail breakdown: {Counter(all_fail_reasons)}")


def process_verilogeval():
    print("\n── VerilogEval ───────────────────────────")
    base = "raw_datasets/verilogeval/dataset_code-complete-iccad2023"
    out_path = "filtered/verilogeval_filtered.jsonl"

    all_files = os.listdir(base)
    ref_files = sorted(f for f in all_files if f.endswith('_ref.sv'))
    print(f"Found {len(ref_files)} ref.sv files")

    done = load_checkpoint("verilogeval")
    candidates = []
    fail_reasons = []

    for ref_fname in tqdm(ref_files, desc="  loading"):
        if ref_fname in done:
            continue

        prefix = ref_fname.replace('_ref.sv', '')
        ref_path = os.path.join(base, ref_fname)
        prompt_path = os.path.join(base, f"{prefix}_prompt.txt")

        with open(ref_path, encoding='utf-8') as f:
            code = f.read()

        instruction = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, encoding='utf-8') as f:
                instruction = f.read().strip()

        if not has_module(code):
            fail_reasons.append("no_module")
            continue

        candidates.append({
            "_key": ref_fname,
            "source": "verilogeval",
            "problem": prefix,
            "instruction": instruction,
            "code": code,
            "standard": "systemverilog"
        })

    new_done = set(done)
    with open(out_path, 'a', encoding='utf-8') as f:
        for ex in candidates:
            key = ex.pop("_key")
            f.write(json.dumps(ex) + '\n')
            new_done.add(key)

    save_checkpoint("verilogeval", new_done)
    print(f"Passed: {len(candidates)} / {len(ref_files)}")
    print(f"Fail breakdown: {Counter(fail_reasons)}")


def process_chisel():
    print("\n── Chisel→Verilog ────────────────────────")

    train_dir = "raw_datasets/chisel_verilog/train"
    out_path = "filtered/chisel_filtered.jsonl"

    arrow_files = sorted(
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith('.arrow')
    )
    print(f"Found {len(arrow_files)} arrow files")

    import pyarrow.ipc as ipc

    done = load_checkpoint("chisel")
    fail_reasons = []
    global_idx = 0

    for arrow_file in arrow_files:
        shard_name = os.path.basename(arrow_file)
        print(f"  Processing shard: {shard_name}")
        candidates = []

        try:
            with open(arrow_file, 'rb') as f:
                reader = ipc.open_file(f)
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)
                    col_names = batch.schema.names
                    for row_i in range(batch.num_rows):
                        key = f"{shard_name}:{global_idx}"
                        global_idx += 1
                        if key in done:
                            continue
                        row = {col: batch.column(col)[row_i].as_py() for col in col_names}
                        instruction = row.get('prompt', '')
                        code = row.get('response', '') or ''
                        if not code or not has_module(code):
                            fail_reasons.append("no_module")
                            continue
                        if not is_verilog_2001(code):
                            fail_reasons.append("systemverilog")
                            continue
                        candidates.append({
                            "_key": key,
                            "source": "chisel_verilog",
                            "instruction": instruction,
                            "code": code
                        })
        except Exception as e:
            print(f"  Skipped {shard_name}: {e}")
            continue

        if candidates:
            print(f"  {len(candidates)} candidates → iverilog...")
            new_done = set(done)
            with open(out_path, 'a', encoding='utf-8') as out_file:
                for i in range(0, len(candidates), CHUNK_SIZE):
                    chunk = candidates[i:i + CHUNK_SIZE]
                    passed, chunk_fail_reasons = filter_chunk(chunk)
                    fail_reasons.extend(chunk_fail_reasons)
                    for ex in passed:
                        key = ex.pop("_key", "")
                        out_file.write(json.dumps(ex) + '\n')
                    out_file.flush()
                    for ex in chunk:
                        new_done.add(ex.get("_key", ""))
                    save_checkpoint("chisel", new_done)
                    print(f"    chunk {i // CHUNK_SIZE + 1}: {len(passed)}/{len(chunk)} passed ✓")
                    gc.collect()
            done = new_done
            candidates = []
            gc.collect()

    print(f"Fail breakdown: {Counter(fail_reasons)}")


def process_mg_verilog():
    print("\n── MG-Verilog ────────────────────────────")

    train_dir = "raw_datasets/mg_verilog/train"
    out_path = "filtered/mg_verilog_filtered.jsonl"

    arrow_files = sorted(
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.endswith('.arrow')
    )
    print(f"Found {len(arrow_files)} arrow files")

    import pyarrow.ipc as ipc

    done = load_checkpoint("mg_verilog")
    fail_reasons = []
    global_idx = 0

    for arrow_file in arrow_files:
        shard_name = os.path.basename(arrow_file)
        print(f"  Processing shard: {shard_name}")
        candidates = []

        try:
            with open(arrow_file, 'rb') as f:
                reader = ipc.open_file(f)
                for batch_idx in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_idx)
                    col_names = batch.schema.names
                    if batch_idx == 0 and global_idx == 0:
                        print(f"  Columns: {col_names}")
                    for row_i in range(batch.num_rows):
                        key = f"{shard_name}:{global_idx}"
                        global_idx += 1
                        if key in done:
                            continue
                        row = {col: batch.column(col)[row_i].as_py() for col in col_names}
                        instruction = (row.get('description') or row.get('prompt') or
                                       row.get('instruction') or '')
                        code = (row.get('code') or row.get('response') or
                                row.get('verilog') or '') or ''
                        if not code or not has_module(code):
                            fail_reasons.append("no_module")
                            continue
                        if not is_verilog_2001(code):
                            fail_reasons.append("systemverilog")
                            continue
                        candidates.append({
                            "_key": key,
                            "source": "mg_verilog",
                            "instruction": instruction,
                            "code": code
                        })
        except Exception as e:
            print(f"  Skipped {shard_name}: {e}")
            continue

        if candidates:
            print(f"  {len(candidates)} candidates → iverilog...")
            new_done = set(done)
            with open(out_path, 'a', encoding='utf-8') as out_file:
                for i in range(0, len(candidates), CHUNK_SIZE):
                    chunk = candidates[i:i + CHUNK_SIZE]
                    passed, chunk_fail_reasons = filter_chunk(chunk)
                    fail_reasons.extend(chunk_fail_reasons)
                    for ex in passed:
                        ex.pop("_key", None)
                        out_file.write(json.dumps(ex) + '\n')
                    out_file.flush()
                    for ex in chunk:
                        new_done.add(ex.get("_key", ""))
                    save_checkpoint("mg_verilog", new_done)
                    print(f"    chunk {i // CHUNK_SIZE + 1}: {len(passed)}/{len(chunk)} passed ✓")
                    gc.collect()
            done = new_done
            candidates = []
            gc.collect()

    print(f"Fail breakdown: {Counter(fail_reasons)}")


# ── Summary ────────────────────────────────────────────────────

def print_summary():
    print("\n\n══ FILTER SUMMARY ════════════════════════")
    total = 0
    for fname in sorted(os.listdir("filtered")):
        if fname.endswith('.jsonl'):
            count = sum(1 for _ in open(f"filtered/{fname}", encoding='utf-8'))
            print(f"  {fname:45s}  {count:>5} examples")
            total += count
    print(f"  {'TOTAL':45s}  {total:>5} examples")
    print()


# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_rtlcoder()
    process_verilogeval()
    process_chisel()
    # process_mg_verilog()
    print_summary()