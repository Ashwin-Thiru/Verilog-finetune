# scripts/cap_debug.py
import json, random
random.seed(42)

with open("synthetic/debug_dataset.jsonl", encoding='utf-8') as f:
    data = [json.loads(l) for l in f]

print(f"Before: {len(data)}")

# But sample smartly — keep diversity across bug types
from collections import defaultdict
by_bug = defaultdict(list)
for item in data:
    by_bug[item['bug_type']].append(item)

# Cap per bug type proportionally — max 1000 per type
targets = {
    'undeclared_signal':     1200,  # still biggest but capped
    'blocking_nonblocking':   600,
    'width_mismatch':         600,
    'missing_begin_end':       60,  # keep all, only 60
    'sensitivity_list':        33,  # keep all, only 33
    'reset_edge':              33,  # keep all, only 33
}

sampled = []
for bug_type, target in targets.items():
    available = by_bug[bug_type]
    n = min(target, len(available))
    sampled.extend(random.sample(available, n))
    print(f"  {bug_type:30s}  {n:4d}")

random.shuffle(sampled)

with open("synthetic/debug_dataset_3k.jsonl", 'w',
          encoding='utf-8') as f:
    for item in sampled:
        f.write(json.dumps(item) + '\n')

print(f"\nAfter: {len(sampled)} debug examples")