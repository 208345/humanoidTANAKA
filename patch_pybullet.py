import os

with open('learning/envs/pybullet_env.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add friction after loading URDF
old_load = '''        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
        )'''
new_load = '''        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
        )
        
        # Set friction to 1.0 for all links
        p.changeDynamics(self._robot_id, -1, lateralFriction=1.0)
        for i in range(p.getNumJoints(self._robot_id)):
            p.changeDynamics(self._robot_id, i, lateralFriction=1.0)'''
content = content.replace(old_load, new_load)

# Add base_linear_velocity to return dict
old_return = '''        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu_quaternion": imu_quaternion,
            "imu_angular_velocity": imu_angular_velocity,
        }'''
new_return = '''        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu_quaternion": imu_quaternion,
            "imu_angular_velocity": imu_angular_velocity,
            "base_linear_velocity": np.array(base_vel, dtype=np.float64),
        }'''
content = content.replace(old_return, new_return)

with open('learning/envs/pybullet_env.py', 'w', encoding='utf-8') as f:
    f.write(content)
