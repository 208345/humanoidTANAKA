"""強化学習環境の基底クラス。

humanoidTANAKA の RobotInterface / Controller 抽象化と同じ思想で、
学習環境もバックエンド（PyBullet / Unity / MuJoCo）を差し替え可能にする。

観測の組み立て方をここに一元化し、PolicyController と共有することで
learning/README.md の最重要注意事項:
「学習時の観測の作り方と、実機での観測の作り方を完全に一致させること」
を構造的に保証する。
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml


def load_robot_params(params_path: str | Path = "model/params.yaml") -> dict:
    """model/params.yaml を読み込む。

    関節数・制御周期など、環境定義に必要なパラメータの唯一の正は
    このファイルである（CONTRIBUTING.md 参照）。
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(
            f"パラメータファイルが見つかりません: {path}\n"
            "Phase 0 で model/params.yaml を用意してください。"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_observation(
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    imu_quaternion: np.ndarray,
    imu_angular_velocity: np.ndarray,
) -> np.ndarray:
    """観測ベクトルを組み立てる。

    この関数を learning/envs/ と control/interface/controller.py (PolicyController)
    の両方から呼ぶことで、観測の不一致を構造的に防止する。

    観測の並び:
        [joint_positions (N), joint_velocities (N),
         imu_quaternion (4), imu_angular_velocity (3)]
    合計: 2*N + 7 次元

    すべて SI 単位 (rad, rad/s)。
    """
    return np.concatenate([
        joint_positions,
        joint_velocities,
        imu_quaternion,
        imu_angular_velocity,
    ]).astype(np.float32)


class HumanoidEnvBase(gym.Env):
    """ヒューマノイド強化学習環境の基底クラス。

    サブクラスは以下を実装する:
        - _sim_reset()    : 物理エンジンを初期化し、初期観測を返す
        - _sim_step()     : 行動を適用して1ステップ進め、観測を返す

    報酬関数は外部から注入可能（rewards.py と分離）。
    """

    metadata = {"render_modes": ["human", "direct"]}

    def __init__(
        self,
        params_path: str | Path = "model/params.yaml",
        reward_fn=None,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.params = load_robot_params(params_path)

        # 関節数は params.yaml から取得
        self.num_joints = self.params["robot"]["num_joints"]
        self.control_hz = self.params["timing"]["control_hz"]
        self.control_dt = 1.0 / self.control_hz

        # 関節リミットの取得（null の場合はデフォルト値を使用）
        joints = self.params["joints"]
        self.joint_names = [j["name"] for j in joints]
        self.torque_limits = np.array([
            j.get("torque_max") or 1.0 for j in joints
        ], dtype=np.float64)

        # --- 行動空間: 関節角の目標値 [rad] ---
        # 正規化された [-1, 1] の指令を受け取り、環境側でスケーリングする
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # --- 観測空間 ---
        # build_observation() の出力に対応: 2*N + 7 次元
        obs_dim = 2 * self.num_joints + 7
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32,
        )

        # 報酬関数（外部注入可能）
        self._reward_fn = reward_fn

        # エピソード管理
        self.max_episode_steps = max_episode_steps
        self._current_step = 0

    def set_reward_fn(self, reward_fn) -> None:
        """報酬関数を差し替える。カリキュラム切替時に使用。"""
        self._reward_fn = reward_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        raw_state = self._sim_reset()
        obs = build_observation(**raw_state)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        raw_state = self._sim_step(action)
        obs = build_observation(**raw_state)

        self._current_step += 1

        # 報酬計算
        if self._reward_fn is not None:
            reward, terminated, info = self._reward_fn(obs, action, raw_state)
        else:
            reward = 0.0
            terminated = False
            info = {}

        truncated = self._current_step >= self.max_episode_steps
        info["is_timeout_success"] = truncated

        return obs, float(reward), terminated, truncated, info

    # --- サブクラスが実装する抽象メソッド ---

    @abstractmethod
    def _sim_reset(self) -> dict:
        """物理エンジンを初期化し、初期状態を返す。

        Returns:
            dict: build_observation() に渡せるキーワード引数。
                  {joint_positions, joint_velocities,
                   imu_quaternion, imu_angular_velocity}
        """

    @abstractmethod
    def _sim_step(self, action: np.ndarray) -> dict:
        """行動を適用して1ステップ進め、状態を返す。

        Args:
            action: [-1, 1] に正規化された関節指令。

        Returns:
            dict: build_observation() に渡せるキーワード引数。
        """

    @abstractmethod
    def close(self) -> None:
        """リソースの解放。"""
