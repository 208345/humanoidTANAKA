import sys

files = ['learning/train/train.py', 'learning/eval/evaluate.py']
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # We know the first two lines are:
    # import os
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    if lines[0].startswith('import os') and lines[1].startswith('os.environ'):
        env_lines = lines[:2]
        rest = lines[2:]
        
        # Find the last __future__ import
        insert_idx = 0
        for i, line in enumerate(rest):
            if line.startswith('from __future__'):
                insert_idx = i + 1
        
        # Insert env_lines after insert_idx
        new_lines = rest[:insert_idx] + env_lines + rest[insert_idx:]
        
        with open(file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

print("Fixed future imports.")
