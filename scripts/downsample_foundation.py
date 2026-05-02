# scripts/downsample_foundation.py
import json
import random
import re
import os

random.seed(42)  # reproducible

# ── Load all foundation data ────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]

rtlcoder  = load_jsonl("filtered/rtlcoder_filtered.jsonl")
chisel    = load_jsonl("filtered/chisel_filtered.jsonl")
verilogeval = load_jsonl("filtered/verilogeval_filtered.jsonl")

print(f"Loaded: RTLCoder={len(rtlcoder)}, "
      f"Chisel={len(chisel)}, "
      f"VerilogEval={len(verilogeval)}")

# ── Topic Classifier ────────────────────────────────────────────

def classify_topic(code, instruction):
    text = (code + " " + instruction).lower()
    
    if any(k in text for k in [
        'fsm', 'state machine', 'state_reg', 
        'next_state', 'current_state'
    ]):
        return 'fsm'
    
    elif any(k in text for k in [
        'fifo', 'queue', 'memory', 'ram', 
        'rom', 'register file'
    ]):
        return 'memory'
    
    elif any(k in text for k in [
        'pipeline', 'stage', 'stall', 
        'flush', 'hazard'
    ]):
        return 'pipeline'
    
    elif any(k in text for k in [
        'uart', 'spi', 'i2c', 'protocol',
        'serial', 'interface'
    ]):
        return 'protocol'
    
    elif any(k in text for k in [
        'counter', 'count', 'clk', 
        'posedge', 'flip.flop', 'register',
        'always @'
    ]):
        return 'sequential'
    
    elif any(k in text for k in [
        'alu', 'adder', 'multiplier', 
        'arithmetic', 'add', 'subtract'
    ]):
        return 'arithmetic'
    
    elif any(k in text for k in [
        'mux', 'decoder', 'encoder', 
        'priority', 'select'
    ]):
        return 'combinational'
    
    else:
        return 'other'

# ── Classify Everything ─────────────────────────────────────────

def classify_all(dataset):
    for item in dataset:
        item['topic'] = classify_topic(
            item.get('code', ''),
            item.get('instruction', '')
        )
    return dataset

rtlcoder    = classify_all(rtlcoder)
chisel      = classify_all(chisel)
verilogeval = classify_all(verilogeval)

# Print topic distribution
from collections import Counter
all_items = rtlcoder + chisel + verilogeval
topic_counts = Counter(item['topic'] for item in all_items)
print("\nTopic distribution across all 19K:")
for topic, count in sorted(topic_counts.items(), 
                            key=lambda x: -x[1]):
    print(f"  {topic:20s}  {count:>6}")

# ── Target Distribution (3000 total) ───────────────────────────

TARGET_TOTAL = 3000

targets = {
    'sequential':    700,   # most important, model struggles
    'fsm':           600,   # most important, model struggles
    'combinational': 400,   # model is OK but needs coverage
    'arithmetic':    300,
    'memory':        300,
    'pipeline':      250,
    'protocol':      200,
    'other':         250,
}

print(f"\nTarget: {sum(targets.values())} examples")

# ── Sample Per Topic ────────────────────────────────────────────

# Group by topic
from collections import defaultdict
by_topic = defaultdict(list)
for item in all_items:
    by_topic[item['topic']].append(item)

sampled = []
for topic, target in targets.items():
    available = by_topic[topic]
    
    if len(available) == 0:
        print(f"  WARNING: No examples for topic '{topic}'")
        continue
    
    # Prefer RTLCoder (has instructions) over Chisel
    # Sort: verilogeval first (highest quality), 
    #       then rtlcoder, then chisel
    priority_order = {'verilogeval': 0, 
                      'rtlcoder': 1, 
                      'chisel_verilog': 2}
    available_sorted = sorted(
        available,
        key=lambda x: priority_order.get(x['source'], 3)
    )
    
    # Take target count, or all if less available
    n = min(target, len(available_sorted))
    
    # Random sample from top 2x candidates to add variety
    pool = available_sorted[:min(target * 2, len(available_sorted))]
    picked = random.sample(pool, n)
    sampled.extend(picked)
    
    print(f"  {topic:20s}  target={target:4d}  "
          f"available={len(available):6d}  picked={n:4d}")

print(f"\nTotal sampled: {len(sampled)}")

# ── Save ────────────────────────────────────────────────────────

os.makedirs("filtered", exist_ok=True)
out_path = "filtered/foundation_3k.jsonl"

random.shuffle(sampled)
with open(out_path, 'w', encoding='utf-8') as f:
    for item in sampled:
        f.write(json.dumps(item) + '\n')

print(f"Saved to {out_path}")

# ── Verify ──────────────────────────────────────────────────────

print("\nFinal topic breakdown:")
final_topics = Counter(item['topic'] for item in sampled)
for topic, count in sorted(final_topics.items(), 
                            key=lambda x: -x[1]):
    pct = count / len(sampled) * 100
    print(f"  {topic:20s}  {count:4d}  ({pct:.1f}%)")