"""
diagnostic.py - Count per-class distribution in training labels
"""
import os
from pathlib import Path
from collections import defaultdict

base = Path('data/labels/train')
class_names = {0: 'defect_1', 1: 'defect_2', 2: 'defect_3', 3: 'defect_4'}

img_count  = defaultdict(int)
inst_count = defaultdict(int)
total_images = 0
empty_images = 0

for f in base.glob('*.txt'):
    total_images += 1
    lines = [l for l in f.read_text().strip().splitlines() if l.strip()]
    if not lines:
        empty_images += 1
        continue
    seen = set()
    for line in lines:
        cls = int(line.split()[0])
        inst_count[cls] += 1
        seen.add(cls)
    for cls in seen:
        img_count[cls] += 1

print(f"Total train label files : {total_images}")
print(f"Empty (background-only) : {empty_images}")
print()
header = f"{'Class':<8} {'Name':<12} {'Images':<10} {'Instances':<12} {'Img %':<10} Inst/Img"
print(header)
print('-' * len(header))
for cls in sorted(class_names):
    imgs  = img_count[cls]
    insts = inst_count[cls]
    pct   = imgs / total_images * 100
    ratio = insts / imgs if imgs else 0
    print(f"{cls:<8} {class_names[cls]:<12} {imgs:<10} {insts:<12} {pct:<10.1f} {ratio:.2f}")

print()
total_insts = sum(inst_count.values())
print(f"Total defect instances  : {total_insts}")
max_cls = max(inst_count, key=inst_count.get)
min_cls = min(inst_count, key=inst_count.get)
ratio = inst_count[max_cls] / max(inst_count[min_cls], 1)
print(f"Imbalance ratio (max/min instances): {ratio:.1f}x")
