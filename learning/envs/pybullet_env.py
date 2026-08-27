"""PyBullet バックエンドの強化学習環境。

URDF ファイルが用意でき次第、urdf_path 引数にパスを渡すだけで動作する。
Unity や MuJoCo に差し替える場合は、base_env.HumanoidEnvBase を
継承して同じインターフェースの別クラスを作ればよい。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.envs.base_env import HumanoidEnvBase


class PyBulletHumanoidEnv(HumanoidEnvBase):
    """PyBullet を物理エンジンとして使用するヒューマノイド環境。

    Args:
        urdf_path: ロボットの URDF ファイルパス。
        params_path: model/params.yaml のパス。
        reward_fn: 報酬関数（None の場合は報酬 0）。
        render_mode: "human" で GUI 表示、None で非表示。
        max_episode_steps: エピソードの最大ステップ数。
        action_scale: [-1,1] の行動を関節角 [rad] に変換するスケール。
        n_substeps: 1制御ステップあたりの物理シミュレーションの細分化数。
    """

    def __init__(
        self,
        urdf_path: str | Path,
        params_path: str | Path = "model/params.yaml",
        reward_fn=None,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        action_scale: float = 0.5,
        n_substeps: int = 4,
    ) -> None:
        # PyBullet の遅延 import（未インストール時に他の環境が動くように）
        try:
            import pybullet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyBullet が見つかりません。'pip install pybullet' を実行してください。"
            ) from exc

        super().__init__(
            params_path=params_path,
            reward_fn=reward_fn,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        self._urdf_path = Path(urdf_path)
        if not self._urdf_path.exists():
            raise FileNotFoundError(
                f"URDF ファイルが見つかりません: {self._urdf_path}\n"
                "model/ ディレクトリに URDF を用意してください。"
            )

        self._action_scale = action_scale
        self._n_substeps = n_substeps

        # PyBullet の初期化
        import pybullet as p
        import pybullet_data

        self._p = p

        if render_mode == "human":
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ロボットと関節のマッピングは _sim_reset() で構築
        self._robot_id = None
        self._controllable_joints = []  # 制御対象の関節インデックス

    def _discover_joints(self) -> list[int]:
        """URDF から制御可能な関節（REVOLUTE / PRISMATIC）を探索する。

        params.yaml の num_joints と数が一致しない場合はエラーにする。
        """
        p = self._p
        controllable = []
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                controllable.append(i)

        if len(controllable) != self.num_joints:
            raise ValueError(
                f"URDF の制御可能関節数 ({len(controllable)}) と "
                f"params.yaml の num_joints ({self.num_joints}) が一致しません。\n"
                f"URDF 関節: {controllable}"
            )
        return controllable

    def _read_state(self) -> dict:
        """PyBullet から現在のセンサ値を読み取る。"""
        p = self._p

        # 関節の状態
        joint_states = p.getJointStates(self._robot_id, self._controllable_joints)
        joint_positions = np.array([s[0] for s in joint_states], dtype=np.float64)
        joint_velocities = np.array([s[1] for s in joint_states], dtype=np.float64)

        # 胴体（ベースリンク）の姿勢と角速度
        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self._robot_id)

        # PyBullet のクォータニオンは (x, y, z, w) だが、
        # RobotState の仕様は (w, x, y, z) なので並び替える
        imu_quaternion = np.array(
            [base_orn[3], base_orn[0], base_orn[1], base_orn[2]], dtype=np.float64
        )
        imu_angular_velocity = np.array(base_ang_vel, dtype=np.float64)

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu_quaternion": imu_quaternion,
            "imu_angular_velocity": imu_angular_velocity,
            "base_linear_velocity": np.array(base_vel, dtype=np.float64),
        }

    def _sim_reset(self) -> dict:
        p = self._p

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # ロボットを少し浮かせた状態で読み込む（足が地面に埋まるのを防ぐ）
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
        )
        
        # Set friction to 1.0 for all links
        p.changeDynamics(self._robot_id, -1, lateralFriction=1.0)
        for i in range(p.getNumJoints(self._robot_id)):
            p.changeDynamics(self._robot_id, i, lateralFriction=1.0)

        self._controllable_joints = self._discover_joints()

        # 全関節の位置制御モーターをオフにしてトルク制御に切り替え
        for idx in self._controllable_joints:
            p.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=p.VELOCITY_CONTROL, force=0,
            )

        # 物理を少し進めて安定させる
        for _ in range(10):
            p.stepSimulation()

        return self._read_state()

    def _sim_step(self, action: np.ndarray) -> dict:
        p = self._p

        # [-1, 1] を関節角目標値にスケーリングして位置制御で適用
        target_positions = action * self._action_scale

        for i, idx in enumerate(self._controllable_joints):
            p.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_positions[i]),
                force=float(self.torque_limits[i]),
                maxVelocity=5.0,
            )

        # サブステップ（物理の細分化）
        physics_dt = 1.0 / 240.0  # PyBullet のデフォルト
        steps_per_control = max(int(round(self.control_dt / physics_dt)), 1)
        for _ in range(steps_per_control):
            p.stepSimulation()

        return self._read_state()

    def render(self):
        # PyBullet の GUI が自動で描画する
        pass

    def close(self) -> None:
        self._p.disconnect(self._physics_client)