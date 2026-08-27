with open('learning/envs/base_env.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_sig = '''def build_observation(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    imu_quaternion: np.ndarray,
    imu_angular_velocity: np.ndarray,
) -> np.ndarray:'''
new_sig = '''def build_observation(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    imu_quaternion: np.ndarray,
    imu_angular_velocity: np.ndarray,
    **kwargs,
) -> np.ndarray:'''

content = content.replace(old_sig, new_sig)

with open('learning/envs/base_env.py', 'w', encoding='utf-8') as f:
    f.write(content)
