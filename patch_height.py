with open('learning/envs/pybullet_env.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('basePosition=[0, 0, 0.5]', 'basePosition=[0, 0, 0.2]')

with open('learning/envs/pybullet_env.py', 'w', encoding='utf-8') as f:
    f.write(content)
