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
    base_linear_velocity: np.ndarray | None = None,
    gait_phase: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """観測ベクトルを組み立てる。

    この関数を learning/envs/ と control/interface/controller.py (PolicyController)
    の両方から呼ぶことで、観測の不一致を構造的に防止する。

    観測の並び:
        [joint_positions (N), joint_velocities (N),
         imu_quaternion (4), imu_angular_velocity (3),
         base_linear_velocity (3), gait_clock (2)]
    合計: 2*N + 12 次元

    gait_clock = [sin(2πφ), cos(2πφ)]。φ∈[0,1) は歩行周期の位相で、
    実機でも制御ループ内で同じ周期で進めること（外部センサは不要）。

    すべて SI 単位 (rad, rad/s, m/s)。base_linear_velocity は歩行の
    速度追従に必須（実機では IMU 積分 or 外部計測で埋める）。
    """
    if base_linear_velocity is None:
        base_linear_velocity = np.zeros(3, dtype=np.float32)
    return np.concatenate([
        joint_positions,
        joint_velocities,
        imu_quaternion,
        imu_angular_velocity,
        base_linear_velocity,
        [np.sin(2.0 * np.pi * gait_phase), np.cos(2.0 * np.pi * gait_phase)],
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
        # 歩行周期 [s]（左右1歩ずつで1周期）。報酬が「今どちらの足を上げるべきか」
        # をこの位相で決めることで、足を高速に震わせて接地を稼ぐ抜け道を塞ぐ。
        self.gait_period = self.params["timing"].get("gait_period", 0.6)
        self._gait_phase = 0.0

        # 参照歩容（残差制御）。params.yaml に reference_gait があると有効:
        #   関節目標角 = 参照歩容(位相) * ランプ + action * residual_scale
        # 行動の意味が変わるので、無効時に学習したポリシーとは互換性が無い。
        self.reference = None
        ref_cfg = self.params.get("reference_gait")
        if ref_cfg:
            ref_path = Path(params_path).parent / Path(ref_cfg["file"]).name
            with open(ref_path, encoding="utf-8") as f:
                self.reference = yaml.safe_load(f)
            self.residual_scales = np.asarray(ref_cfg["residual_scale"], dtype=np.float64)
            # 参照歩容の周期で位相時計を回す（観測の clock と参照の位相を一致させる）
            self.gait_period = float(self.reference["T"])

        # 関節リミットの取得（null の場合はデフォルト値を使用）
        joints = self.params["joints"]
        self.joint_names = [j["name"] for j in joints]
        self.torque_limits = np.array([
            j.get("torque_max") or 1.0 for j in joints
        ], dtype=np.float64)

        # 可動域 [rad]（POSITION_CONTROL の目標値をここでクリップする）
        self.joint_lower = np.array([
            j.get("limit_lower", -np.pi) for j in joints
        ], dtype=np.float64)
        self.joint_upper = np.array([
            j.get("limit_upper", np.pi) for j in joints
        ], dtype=np.float64)

        # 関節ごとの行動スケール: [-1,1] の指令を関節角 [rad] に変換する係数。
        # params.yaml の各関節に action_scale があれば使い、無ければ 0.5（従来の一律値）。
        # 膝は歩行で大きく曲げる必要があるため他より広く取る。
        self.action_scales = np.array([
            j.get("action_scale", 0.5) for j in joints
        ], dtype=np.float64)

        # --- 行動空間: 関節角の目標値 [rad] ---
        # 正規化された [-1, 1] の指令を受け取り、環境側でスケーリングする
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # --- 観測空間 ---
        # build_observation() の出力に対応: 2*N + 12 次元
        # (joint_pos N, joint_vel N, quat 4, ang_vel 3, base_lin_vel 3, gait_clock 2)
        obs_dim = 2 * self.num_joints + 12
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high, high=high, dtype=np.float32,
        )

        # 報酬関数（外部注入可能）
        self._reward_fn = reward_fn

        # エピソード管理
        self.max_episode_steps = max_episode_steps
        self._current_step = 0

    def reference_targets(self, phase: float, step: int) -> np.ndarray:
        """参照歩容の関節目標角 [rad]（model/reference_gait.yaml の式）。開始 0.5s でランプ。"""
        g = self.reference
        wt = 2.0 * np.pi * phase
        out = np.zeros(self.num_joints)
        for leg, (s, m, th) in enumerate([(1, 1, 0.0), (-1, -1, np.pi)]):
            o = leg * 4
            out[o + 0] = s * g["r_add"] + g["r_amp"] * np.sin(wt + g["phr"])
            out[o + 1] = m * (g["p_off"] + g["p_amp"] * np.sin(wt + th))
            out[o + 2] = m * (g["k_off"] + g["k_amp"] * max(0.0, np.sin(wt + th + g["phk"])))
            out[o + 3] = m * (g["a_off"] + g["a_amp"] * np.sin(wt + th + g["pha"]))
        ramp = min(1.0, step * self.control_dt / 0.5)
        return np.clip(out, -1.5, 1.5) * ramp

    def set_reward_fn(self, reward_fn) -> None:
        """報酬関数を差し替える。カリキュラム切替時に使用。"""
        self._reward_fn = reward_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        # 開始位相はランダム（どちらの足から踏み出しても歩けるように）
        self._gait_phase = float(self.np_random.uniform(0.0, 1.0))
        raw_state = self._sim_reset()
        raw_state["gait_phase"] = self._gait_phase
        obs = build_observation(**raw_state)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        raw_state = self._sim_step(action)
        self._gait_phase = (self._gait_phase + self.control_dt / self.gait_period) % 1.0
        raw_state["gait_phase"] = self._gait_phase
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
