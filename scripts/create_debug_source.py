# scripts/create_debug_source.py
import json
import random

random.seed(99)  # different seed from foundation_3k

# Load ALL filtered data
def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]

rtlcoder    = load_jsonl("filtered/rtlcoder_filtered.jsonl")
chisel      = load_jsonl("filtered/chisel_filtered.jsonl")
verilogeval = load_jsonl("filtered/verilogeval_filtered.jsonl")

all_data = rtlcoder + chisel + verilogeval

# Load what's already used in foundation_3k
used = load_jsonl("filtered/foundation_3k.jsonl")

# Get codes already used (to avoid overlap)
used_codes = set(
    item['code'] for item in used
)

# Filter out already used modules
remaining = [
    item for item in all_data
    if item.get('code', '') not in used_codes
]

print(f"Total available:  {len(all_data)}")
print(f"Used in codegen:  {len(used_codes)}")
print(f"Remaining unused: {len(remaining)}")

# Sample 3,000 from remaining for debug source
# Prefer sequential and FSM since those are your weak spots
from collections import defaultdict, Counter

# Classify topics (reuse same classifier)
SV_KEYWORDS = ['always_ff','always_comb','logic ']

def classify_topic(code, instruction):
    text = (code + " " + instruction).lower()
    if any(k in text for k in ['fsm','state machine','next_state']):
        return 'fsm'
    elif any(k in text for k in ['counter','posedge','flip','register']):
        return 'sequential'
    elif any(k in text for k in ['fifo','memory','ram','rom']):
        return 'memory'
    elif any(k in text for k in ['pipeline','stage','stall']):
        return 'pipeline'
    elif any(k in text for k in ['uart','spi','i2c','serial']):
        return 'protocol'
    elif any(k in text for k in ['alu','adder','multiplier','arithmetic']):
        return 'arithmetic'
    elif any(k in text for k in ['mux','decoder','encoder']):
        return 'combinational'
    else:
        return 'other'

for item in remaining:
    item['topic'] = classify_topic(
        item.get('code',''),
        item.get('instruction','')
    )

# Group by topic
by_topic = defaultdict(list)
for item in remaining:
    by_topic[item['topic']].append(item)

print("\nAvailable topics in remaining pool:")
for t, items in sorted(by_topic.items(),
                        key=lambda x: -len(x[1])):
    print(f"  {t:20s}  {len(items)}")

# Sample with heavy weight on sequential/FSM
# because those are the ones that fail most in debugging
targets = {
    'sequential':    900,
    'fsm':           700,
    'memory':        300,
    'arithmetic':    300,
    'pipeline':      250,
    'combinational': 200,
    'protocol':      150,
    'other':         200,
}

sampled = []
for topic, target in targets.items():
    available = by_topic[topic]
    n = min(target, len(available))
    sampled.extend(random.sample(available, n))
    print(f"  {topic:20s}  picked {n}/{target}")

print(f"\nTotal debug source: {len(sampled)}")

# Save
random.shuffle(sampled)
with open("filtered/debug_source.jsonl", 'w',
          encoding='utf-8') as f:
    for item in sampled:
        f.write(json.dumps(item) + '\n')

print("Saved to filtered/debug_source.jsonl")