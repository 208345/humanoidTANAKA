with open('learning/envs/unity_env.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_return = '''        return {
            "joint_positions": obs_array[0:n],
            "joint_velocities": obs_array[n:2*n],
            "imu_quaternion": obs_array[2*n:2*n+4],
            "imu_angular_velocity": obs_array[2*n+4:2*n+7]
        }'''
new_return = '''        return {
            "joint_positions": obs_array[0:n],
            "joint_velocities": obs_array[n:2*n],
            "imu_quaternion": obs_array[2*n:2*n+4],
            "imu_angular_velocity": obs_array[2*n+4:2*n+7],
            "base_linear_velocity": __import__('numpy').array([0.0, 0.0, 0.0], dtype=__import__('numpy').float64)
        }'''
content = content.replace(old_return, new_return)

with open('learning/envs/unity_env.py', 'w', encoding='utf-8') as f:
    f.write(content)
