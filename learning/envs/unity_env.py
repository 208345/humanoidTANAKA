"""Unity (ML-Agents) 環境ラッパー。"""
import numpy as np
import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from learning.envs.base_env import HumanoidEnvBase, load_robot_params

class UnityHumanoidEnv(HumanoidEnvBase):
    def __init__(
        self,
        urdf_path: str,
        params_path: str,
        reward_fn=None,
        render_mode: str | None = None,
        file_name: str | None = None,
    ):
        super().__init__(max_episode_steps=1000, reward_fn=reward_fn)
        self.params = load_robot_params(params_path)
        self.num_joints = self.params["robot"]["num_joints"]
        
        self._env = UnityEnvironment(file_name=file_name, no_graphics=(render_mode!="human"))
        self._env.reset()
        
        self.behavior_name = list(self._env.behavior_specs.keys())[0]

    def _sim_reset(self) -> dict:
        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        
        if len(decision_steps) == 0:
            obs_array = np.zeros(19, dtype=np.float32)
        else:
            obs_array = decision_steps.obs[0][0]
            
        return self._parse_obs(obs_array)

    def _sim_step(self, action: np.ndarray) -> dict:
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action.reshape(1, -1))
        
        self._env.set_actions(self.behavior_name, action_tuple)
        self._env.step()
        
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        
        if len(terminal_steps) > 0:
            obs_array = terminal_steps.obs[0][0]
        elif len(decision_steps) > 0:
            obs_array = decision_steps.obs[0][0]
        else:
            obs_array = np.zeros(19, dtype=np.float32)
            
        return self._parse_obs(obs_array)

    def _parse_obs(self, obs_array: np.ndarray) -> dict:
        n = self.num_joints
        return {
            "joint_positions": obs_array[0:n],
            "joint_velocities": obs_array[n:2*n],
            "imu_quaternion": obs_array[2*n:2*n+4],
            "imu_angular_velocity": obs_array[2*n+4:2*n+7]
        }

    def close(self):
        self._env.close()
