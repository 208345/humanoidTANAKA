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
        target_velocity: float = 0.2,
        velocity_weight: float = 5.0,
        survival_reward: float = 6.0,
        fall_penalty: float = -80.0,
        fall_threshold: float = 1.0,
        orientation_weight: float = 8.0,
        ang_vel_weight: float = 0.05,
        effort_weight: float = 0.002,
        smoothness_weight: float = 0.03,
        lateral_weight: float = 8.0,
        heading_weight: float = 9.0,
        alternation_weight: float = 3.0,
        double_support_penalty: float = 3.0,
        swing_weight: float = 100.0,
        slip_weight: float = 8.0,
        foot_rest_height: float = 0.014,   # 全関節0で接地時の足リンク原点z（8DOFモデル実測）
        swing_clear_cap: float = 0.05,
        lift_vel_weight: float = 3.0,
        skim_penalty: float = 22.0,
        skim_target: float = 0.05,
        height_weight: float = 35.0,
        target_height: float = 0.172,
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
        self.lateral_weight = lateral_weight      # 横ズレ(vy)の抑制
        self.heading_weight = heading_weight      # 向き(yaw)を前方に保つ抑制
        # 「両足べったり接地したまま滑走する」を防ぎ、片足支持の交互歩行を
        # 直接要求するための項。effort/smoothness を弱めたのもこれとセット
        # （動かないほど得、という抜け道を塞ぐ）。
        self.alternation_weight = alternation_weight
        self.double_support_penalty = double_support_penalty
        self.swing_weight = swing_weight            # 遊脚を持ち上げる報酬
        self.slip_weight = slip_weight              # 接地中の足が滑ることへのペナルティ
        self.foot_rest_height = foot_rest_height    # 接地時の足リンク原点の高さ(基準)
        self.swing_clear_cap = swing_clear_cap      # クリアランス報酬の頭打ち
        self.lift_vel_weight = lift_vel_weight      # 遊脚が上向きに動くこと自体を加点
        self.skim_penalty = skim_penalty           # 遊脚が地面すれすれ = すり足へのペナルティ
        self.skim_target = skim_target             # この高さ未満の遊脚を減点
        self.height_weight = height_weight          # 胴体を低くしすぎない
        self.target_height = target_height
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

        # 進行方向(yaw)。スポーン時の向き(0 rad)を「まっすぐ前」の基準とする。
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        terminated = tilt_angle > self.fall_threshold

        reward = self.survival_reward

        # 1. 姿勢報酬
        reward += up_z * self.orientation_weight

        # 2. 速度追従報酬（前進速度 vx が目標に近いほど高い）
        base_vel = raw_state.get("base_linear_velocity", None)
        if base_vel is None:
            base_vel = np.zeros(3)
        base_vel = np.asarray(base_vel, dtype=np.float64)
        forward_vel = float(base_vel[0])        # x = 前方向
        lateral_vel = float(base_vel[1])        # y = 横方向

        vel_err = forward_vel - self.target_velocity
        vel_reward = math.exp(-4.0 * vel_err * vel_err)       # 0〜1
        reward += vel_reward * self.velocity_weight

        # 学習初期のブートストラップ: とにかく前進していれば少し加点
        reward += max(min(forward_vel, self.target_velocity), 0.0) * self.velocity_weight

        # 横滑り・後退のペナルティ（弧を描いて進むのを防ぐ）
        reward -= abs(lateral_vel) * self.lateral_weight
        if forward_vel < 0.0:
            reward += forward_vel * 2.0        # 後退はさらに減点

        # 向きのブレのペナルティ（yaw が 0 からずれる = 曲がって進んでいる）
        reward -= (yaw ** 2) * self.heading_weight

        # 胴体を低くしすぎない（前かがみで腰を落とす姿勢を抑える）
        base_height = raw_state.get("base_height")
        if base_height is not None:
            reward -= max(0.0, self.target_height - float(base_height)) * self.height_weight

        # 2.5 本物の交互ステップを要求する（"こすり滑り" 対策の本命）。
        # foot_contacts / foot_heights / foot_slip は pybullet 環境からのみ供給。
        foot_contacts = raw_state.get("foot_contacts")
        foot_heights = raw_state.get("foot_heights")
        foot_slip = raw_state.get("foot_slip")
        foot_vz = raw_state.get("foot_vz")
        single_support = 0.0
        if foot_contacts is not None and len(foot_contacts) == 2:
            fc = np.asarray(foot_contacts, dtype=np.float64)
            both_down = float(fc[0] * fc[1])
            neither_down = float((1.0 - fc[0]) * (1.0 - fc[1]))
            single_support = 1.0 - both_down - neither_down  # 片足だけ接地なら1

            reward += single_support * self.alternation_weight
            reward -= both_down * self.double_support_penalty

            # (a) 接地している足が水平に動く = 地面をこすっている → 強くペナルティ。
            #     きちんと踏みしめた軸足は速度ゼロのはず。
            if foot_slip is not None:
                fs = np.asarray(foot_slip, dtype=np.float64)
                reward -= float(np.sum(fc * fs)) * self.slip_weight

            # (b) 接地していない足（遊脚）を、基準高さから実際に持ち上げた分だけ加点。
            if foot_heights is not None:
                fh = np.asarray(foot_heights, dtype=np.float64)
                clear = np.clip(fh - self.foot_rest_height, 0.0, self.swing_clear_cap)
                reward += float(np.sum((1.0 - fc) * clear)) * self.swing_weight

            # (c) 遊脚が「上向きに動いている」こと自体を加点。
            #     高さが出る前の段階でも勾配が付くので、"足を上げない" 局所解を抜けやすい。
            if foot_vz is not None:
                fv = np.asarray(foot_vz, dtype=np.float64)
                reward += float(np.sum((1.0 - fc) * np.clip(fv, 0.0, 0.5))) * self.lift_vel_weight

            # (d) 遊脚が skim_target 未満 = すり足。これを積極的に損にする
            #     （"上げたら加点" だけでは倒れるリスクに勝てないため、下限を罰する）。
            if foot_heights is not None:
                fh = np.asarray(foot_heights, dtype=np.float64)
                deficit = np.clip(self.skim_target - (fh - self.foot_rest_height), 0.0, self.skim_target)
                reward -= float(np.sum((1.0 - fc) * deficit)) * self.skim_penalty

        alternation_bonus = single_support

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

        contact_slip = 0.0
        if foot_contacts is not None and foot_slip is not None:
            fc = np.asarray(foot_contacts, dtype=np.float64)
            fs = np.asarray(foot_slip, dtype=np.float64)
            denom = float(np.sum(fc))
            contact_slip = float(np.sum(fc * fs) / denom) if denom > 0 else 0.0

        info = {"tilt_angle": tilt_angle, "up_z": up_z, "yaw": yaw,
                "forward_vel": forward_vel, "lateral_vel": lateral_vel,
                "vel_reward": vel_reward, "single_support": alternation_bonus,
                "contact_slip": contact_slip,
                "base_height": float(base_height) if base_height is not None else 0.0}
        return reward, terminated, info