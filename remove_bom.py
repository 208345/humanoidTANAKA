import sys

files = ['learning/train/train.py', 'learning/eval/evaluate.py']
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove BOM if present
    content = content.replace('\ufeff', '')
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print("BOM removed.")
