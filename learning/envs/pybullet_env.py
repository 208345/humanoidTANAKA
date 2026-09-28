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
        action_scale: params.yaml に per-joint の action_scale が無い関節に
                      使うフォールバックのスカラー値。
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

        # base_env が params.yaml から self.action_scales(per-joint)を構築済み。
        # params に action_scale が無い関節はデフォルト 0.5 になっているので、
        # コンストラクタ引数で明示指定された場合のみそのスカラーで上書きする。
        if action_scale != 0.5:
            self.action_scales[:] = action_scale
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

        # ワールドは初回 _sim_reset() で一度だけ構築し、以降は
        # saveState/restoreState で復元する（resetSimulation+loadURDF は
        # 凹メッシュの衝突形状生成で ~90ms かかり、短いエピソードだと
        # 実行時間の大半をリセットが占めて並列化が効かなくなるため）。
        self._robot_id = None
        self._controllable_joints = []  # 制御対象の関節インデックス
        self._world_built = False
        self._reset_state_id = -1

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

        # 各足の接地判定・高さ・水平すべり速度。
        # 「地面をこすりながら滑走する」を報酬側で直接ペナルティにするために使う。
        foot_contacts = np.zeros(len(self._foot_links), dtype=np.float64)
        foot_heights = np.zeros(len(self._foot_links), dtype=np.float64)
        foot_slip = np.zeros(len(self._foot_links), dtype=np.float64)
        foot_vz = np.zeros(len(self._foot_links), dtype=np.float64)
        foot_vy = np.zeros(len(self._foot_links), dtype=np.float64)
        for i, link in enumerate(self._foot_links):
            contacts = p.getContactPoints(bodyA=self._robot_id, bodyB=self._plane_id, linkIndexA=link)
            foot_contacts[i] = 1.0 if len(contacts) > 0 else 0.0
            ls = p.getLinkState(self._robot_id, link, computeLinkVelocity=1)
            foot_heights[i] = ls[0][2]
            foot_slip[i] = float(np.hypot(ls[6][0], ls[6][1]))  # 水平方向の速さ [m/s]
            foot_vz[i] = float(ls[6][2])                          # 上下方向の速度 [m/s]
            foot_vy[i] = float(ls[6][1])                          # 前方向(+y)の速度 [m/s]

        # 足以外（胴体・脛・膝など）が地面に触れているか。座り込み/膝つき歩行の検出用。
        body_contact = 0.0
        for c in p.getContactPoints(bodyA=self._robot_id, bodyB=self._plane_id):
            if c[3] not in self._foot_links:
                body_contact = 1.0
                break

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "imu_quaternion": imu_quaternion,
            "imu_angular_velocity": imu_angular_velocity,
            "base_linear_velocity": np.array(base_vel, dtype=np.float64),
            "base_height": float(base_pos[2]),
            "foot_contacts": foot_contacts,
            "foot_heights": foot_heights,
            "foot_slip": foot_slip,
            "foot_vz": foot_vz,
            "foot_vy": foot_vy,
            "body_contact": body_contact,
        }

    def _build_world(self) -> None:
        """ワールドを一度だけ構築する（プレーン + ロボット + 動力学設定）。

        以降のエピソードはこの状態を saveState/restoreState で復元するので、
        ここは初回のみ実行される。
        """
        p = self._p

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._plane_id = p.loadURDF("plane.urdf")

        # ロボットを少し浮かせた状態で読み込む（足が地面に埋まるのを防ぐ）
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
        )

        # 摩擦: 地面と胴体は標準的な値。
        p.changeDynamics(self._plane_id, -1, lateralFriction=1.0, restitution=0.0)
        p.changeDynamics(self._robot_id, -1, lateralFriction=1.0)
        for i in range(p.getNumJoints(self._robot_id)):
            p.changeDynamics(self._robot_id, i, lateralFriction=1.0)

        self._controllable_joints = self._discover_joints()

        # 足リンク（各脚チェーンの末端）を名前で特定する。
        self._foot_links = [
            idx for idx in self._controllable_joints
            if p.getJointInfo(self._robot_id, idx)[12].decode("utf-8")
            in ("LU4_1", "RU4_1")
        ]

        # 足だけしっかりグリップさせる（踏みしめる動きを物理的に有利にし、
        # こすり滑る動きを不利にする）。
        for link in self._foot_links:
            p.changeDynamics(
                self._robot_id, link,
                lateralFriction=2.0,
                spinningFriction=0.1,
                rollingFriction=0.01,
                frictionAnchor=1,
                restitution=0.0,
            )

        self._world_built = True

    def _sim_reset(self) -> dict:
        p = self._p

        if not self._world_built:
            self._build_world()
            # 「関節0・原点で浮遊・静止」の綺麗な状態をスナップショットする。
            self._reset_state_id = p.saveState()
        else:
            # 90ms の resetSimulation+loadURDF を ~1ms の復元で置き換える。
            p.restoreState(self._reset_state_id)

        # モーターOFF（トルク制御）を再アサート。restoreState では戻らないため。
        for idx in self._controllable_joints:
            p.setJointMotorControl2(
                self._robot_id, idx,
                controlMode=p.VELOCITY_CONTROL, force=0,
            )

        # 初期姿勢: 全関節ほぼ 0（直立）＋わずかなランダムノイズ。
        # ノイズは探索を促し、実機の個体差にも頑健にする。
        rng = getattr(self, "np_random", None)
        for idx in self._controllable_joints:
            noise = float(rng.uniform(-0.03, 0.03)) if rng is not None else 0.0
            p.resetJointState(self._robot_id, idx, targetValue=noise, targetVelocity=0.0)

        # 足裏が地面すれすれに来るようベース高さを合わせる（0.5m から落として
        # 叩きつけると毎回転倒するため）。全リンクの AABB 最下点を求めて補正。
        # 最下点(=足裏)を地面から 3mm だけ浮かせる: 0 ぴったりだと初期貫入で
        # 弾かれ、逆に高すぎると落下の初速で姿勢が乱れる。
        p.performCollisionDetection()  # getAABB を現在の関節角で確定させる
        link_ids = [-1] + list(range(p.getNumJoints(self._robot_id)))
        z_min = min(p.getAABB(self._robot_id, li)[0][2] for li in link_ids)
        pos, orn = p.getBasePositionAndOrientation(self._robot_id)
        p.resetBasePositionAndOrientation(
            self._robot_id,
            [pos[0], pos[1], pos[2] - z_min + 0.003],
            orn,
        )
        p.resetBaseVelocity(self._robot_id, [0, 0, 0], [0, 0, 0])

        # 物理を少し進めて安定させる
        for _ in range(10):
            p.stepSimulation()

        return self._read_state()

    def _sim_step(self, action: np.ndarray) -> dict:
        p = self._p

        # [-1, 1] を関節ごとのスケールで関節角へ変換し、可動域にクリップ。
        # 参照歩容が有効なら action は参照からの残差。
        if self.reference is not None:
            base = self.reference_targets(self._gait_phase, self._current_step)
            target_positions = base + action * self.residual_scales
        else:
            target_positions = action * self.action_scales
        target_positions = np.clip(target_positions, self.joint_lower, self.joint_upper)

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