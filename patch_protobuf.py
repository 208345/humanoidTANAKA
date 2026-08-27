import sys

files = ['learning/train/train.py', 'learning/eval/evaluate.py']
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION' not in content:
        content = "import os\nos.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'\n" + content
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)

print("Patched.")
