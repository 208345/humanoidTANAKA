"""報酬関数の定義。

環境クラスから分離することで、カリキュラム学習時に報酬関数だけを
差し替えたり、異なるタスク（直立、歩行、片足立ち）で使い分けられる。

cube-sim-rl の知見:
    - 高い生存報酬で早期終了方策を防止
    - ハイブリッド・バリア関数で限界接近を拒絶
    - 終了ペナルティは生存報酬の 30-50 倍
をヒューマノイド向けに拡張。
"""

from __future__ import annotations

import math

import numpy as np


class StandingReward:
    """直立維持の報酬関数（Phase 3 向け）。

    ヒューマノイドが倒れずに立ち続けることを学習させる。

    Args:
        num_joints: 関節数（観測ベクトルの解釈に必要）。
        survival_reward: 生存報酬（毎ステップの基本報酬）。
        fall_penalty: 転倒時のペナルティ。
        fall_threshold: 転倒と判定する胴体の傾き [rad]。
        height_target: 目標とする胴体高さ [m]（URDF に合わせて調整）。
        height_weight: 高さ報酬の重み。
        orientation_weight: 姿勢報酬の重み。
        velocity_weight: 角速度ペナルティの重み（振動抑制）。
        effort_weight: 制御入力ペナルティの重み（省エネ）。
        barrier_power: バリア関数の指数（4 推奨）。
        barrier_weight: バリア関数の重み。
    """

    def __init__(
        self,
        num_joints: int = 12,
        survival_reward: float = 5.0,
        fall_penalty: float = -200.0,
        fall_threshold: float = 1.0,
        height_target: float = 0.3,
        height_weight: float = 2.0,
        orientation_weight: float = 5.0,
        velocity_weight: float = 0.1,
        effort_weight: float = 0.01,
        barrier_power: int = 4,
        barrier_weight: float = 20.0,
    ) -> None:
        self.num_joints = num_joints
        self.survival_reward = survival_reward
        self.fall_penalty = fall_penalty
        self.fall_threshold = fall_threshold
        self.height_target = height_target
        self.height_weight = height_weight
        self.orientation_weight = orientation_weight
        self.velocity_weight = velocity_weight
        self.effort_weight = effort_weight
        self.barrier_power = barrier_power
        self.barrier_weight = barrier_weight

    def __call__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        raw_state: dict,
    ) -> tuple[float, bool, dict]:
        """報酬を計算する。

        Args:
            obs: build_observation() の出力。
            action: [-1, 1] に正規化された行動。
            raw_state: 生のセンサ値辞書。

        Returns:
            (reward, terminated, info) のタプル。
        """
        n = self.num_joints

        # 観測ベクトルを分解
        # [joint_pos(N), joint_vel(N), quat(4), ang_vel(3)]
        imu_quat = obs[2 * n: 2 * n + 4]    # (w, x, y, z)
        ang_vel = obs[2 * n + 4: 2 * n + 7]

        # --- 胴体の傾きを計算 ---
        # クォータニオン (w, x, y, z) から、直立（z軸上向き）からの傾き角を算出
        w, x, y, z = imu_quat
        # 胴体の上方向ベクトル（クォータニオンで [0,0,1] を回転）
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt_angle = math.acos(max(min(float(up_z), 1.0), -1.0))

        # --- 転倒判定 ---
        terminated = tilt_angle > self.fall_threshold

        # --- 報酬計算 ---
        reward = self.survival_reward

        # 1. 姿勢報酬（直立に近いほど高い）
        orientation_reward = up_z * self.orientation_weight
        reward += orientation_reward

        # 2. バリア関数ペナルティ（傾きが限界に近づくほど急増）
        if self.fall_threshold > 0:
            tilt_ratio = tilt_angle / self.fall_threshold
            barrier = (tilt_ratio ** self.barrier_power) * self.barrier_weight
            reward -= barrier

        # 3. 角速度ペナルティ（振動抑制）
        ang_vel_penalty = float(np.sum(ang_vel ** 2)) * self.velocity_weight
        reward -= ang_vel_penalty

        # 4. 制御入力ペナルティ（省エネ・滑らかな動き）
        effort_penalty = float(np.sum(action ** 2)) * self.effort_weight
        reward -= effort_penalty

        # 5. 転倒ペナルティ
        if terminated:
            reward += self.fall_penalty

        info = {
            "tilt_angle": tilt_angle,
            "up_z": up_z,
            "orientation_reward": orientation_reward,
            "ang_vel_penalty": ang_vel_penalty,
            "effort_penalty": effort_penalty,
        }

        return reward, terminated, info


class WalkingReward:
    """速度追従歩行の報酬関数（Phase 4 向け）。

    指定された目標速度で前進することを学習させる。

    Args:
        num_joints: 関節数。
        target_velocity: 目標前進速度 [m/s]。
        velocity_weight: 速度追従報酬の重み。
        その他: StandingReward と同様。
    """

    def __init__(
        self,
        num_joints: int = 12,
        target_velocity: float = 0.3,
        velocity_weight: float = 5.0,
        survival_reward: float = 3.0,
        fall_penalty: float = -200.0,
        fall_threshold: float = 1.0,
        orientation_weight: float = 3.0,
        ang_vel_weight: float = 0.05,
        effort_weight: float = 0.005,
        smoothness_weight: float = 0.1,
    ) -> None:
        self.num_joints = num_joints
        self.target_velocity = target_velocity
        self.velocity_weight = velocity_weight
        self.survival_reward = survival_reward
        self.fall_penalty = fall_penalty
        self.fall_threshold = fall_threshold
        self.orientation_weight = orientation_weight
        self.ang_vel_weight = ang_vel_weight
        self.effort_weight = effort_weight
        self.smoothness_weight = smoothness_weight
        self._prev_action = None

    def __call__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        raw_state: dict,
    ) -> tuple[float, bool, dict]:
        n = self.num_joints

        imu_quat = obs[2 * n: 2 * n + 4]
        ang_vel = obs[2 * n + 4: 2 * n + 7]

        w, x, y, z = imu_quat
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt_angle = math.acos(max(min(float(up_z), 1.0), -1.0))

        terminated = tilt_angle > self.fall_threshold

        reward = self.survival_reward

        # 1. 姿勢報酬
        reward += up_z * self.orientation_weight

        # 2. 速度追従報酬（目標速度との差が小さいほど高報酬）
        # raw_state に base_velocity が含まれている場合に使用
        # 含まれていない場合は角速度から推定
        # TODO: PyBullet の getBaseVelocity() を raw_state に追加する
        # 暫定的に角速度の x 成分（前進方向の回転）を代理指標とする

        # 3. 角速度ペナルティ
        reward -= float(np.sum(ang_vel ** 2)) * self.ang_vel_weight

        # 4. 制御入力ペナルティ
        reward -= float(np.sum(action ** 2)) * self.effort_weight

        # 5. 動作の滑らかさ（前ステップとの行動差をペナルティ）
        if self._prev_action is not None:
            smoothness = float(np.sum((action - self._prev_action) ** 2))
            reward -= smoothness * self.smoothness_weight
        self._prev_action = action.copy()

        # 6. 転倒ペナルティ
        if terminated:
            reward += self.fall_penalty
            self._prev_action = None

        info = {"tilt_angle": tilt_angle, "up_z": up_z}
        return reward, terminated, info