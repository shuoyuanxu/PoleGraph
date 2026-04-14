#!/usr/bin/env python3
import csv
import numpy as np

# Load poles
poles = []
with open('pole_map.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x, y = float(row['x']), float(row['y'])
        if -20 <= x <= 20 and -20 <= y <= 20:
            poles.append((x, y))

print(f"Total poles in crop region: {len(poles)}")
if poles:
    xs = [p[0] for p in poles]
    ys = [p[1] for p in poles]
    print(f"X range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"Y range: [{min(ys):.1f}, {max(ys):.1f}]")
    print(f"First 5 poles: {poles[:5]}")
