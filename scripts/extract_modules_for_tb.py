# scripts/extract_modules_for_tb.py
import json
import random

random.seed(42)

with open("filtered/foundation_3k.jsonl", 
          encoding='utf-8') as f:
    data = [json.loads(l) for l in f]

# Priority order for testbench generation
priority_topics = ['fsm', 'sequential', 'memory', 
                   'pipeline', 'arithmetic']

selected = []
for topic in priority_topics:
    modules = [d for d in data if d.get('topic') == topic]
    # Take up to 80 per topic
    take = min(80, len(modules))
    selected.extend(random.sample(modules, take))

# Fill remaining with other topics
remaining_needed = 400 - len(selected)
others = [d for d in data 
          if d.get('topic') not in priority_topics]
selected.extend(random.sample(
    others, min(remaining_needed, len(others))
))

print(f"Selected {len(selected)} modules for testbench gen")

# Save individual text files — easier to copy into Gemini
import os
os.makedirs("testbench_source", exist_ok=True)

# Save in batches of 5
batch_size = 5
for i in range(0, len(selected), batch_size):
    batch = selected[i:i+batch_size]
    batch_num = (i // batch_size) + 1
    
    out_path = f"testbench_source/batch_{batch_num:03d}.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        for j, item in enumerate(batch):
            f.write(f"MODULE {j+1}:\n")
            f.write(f"Topic: {item.get('topic','')}\n")
            f.write("```verilog\n")
            f.write(item.get('code', ''))
            f.write("\n```\n\n")
    
print(f"Saved {batch_num} batch files to testbench_source/")
print(f"Each batch has 5 modules")
print(f"Paste each batch into Gemini with the prompt above")