"""MuJoCo シミュレータ側の実装。

mujoco は重い依存なので、import はコンストラクタ内で行う（遅延 import）。
これにより MuJoCo 未インストールでも他のバックエンドとテストが動く。

Phase 1 の課題: このクラスを完成させ、全関節が指令通り動くことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from control.interface.robot_interface import RobotInterface, RobotState


class SimRobot(RobotInterface):
    def __init__(
        self,
        model_path: str | Path = "model/humanoid.xml",
        control_dt: float = 0.02,
    ) -> None:
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MuJoCo が見つかりません。'pip install -e \".[sim]\"' を実行してください。"
            ) from exc

        import mujoco

        self._mujoco = mujoco
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"モデルファイルがありません: {self._model_path}\n"
                "Phase 0 で model/humanoid.xml を用意してください。"
            )
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._data = mujoco.MjData(self._model)
        self._control_dt = control_dt
        # 制御1ステップあたり何回物理を進めるか
        self._n_substeps = max(int(round(control_dt / self._model.opt.timestep)), 1)

    @property
    def num_joints(self) -> int:
        return self._model.nu

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def reset(self) -> RobotState:
        self._mujoco.mj_resetData(self._model, self._data)
        self._mujoco.mj_forward(self._model, self._data)
        return self.read_state()

    def read_state(self) -> RobotState:
        d = self._data
        # NOTE: 自由浮遊のベースがある場合、qpos の先頭7要素は
        #       ベースの位置(3) + クォータニオン(4) なので関節はその後ろ。
        #       model/humanoid.xml の構成に合わせてここを調整すること。
        n = self.num_joints
        return RobotState(
            joint_positions=d.qpos[-n:].copy(),
            joint_velocities=d.qvel[-n:].copy(),
            imu_quaternion=d.qpos[3:7].copy(),
            imu_angular_velocity=d.qvel[3:6].copy(),
            imu_linear_acceleration=np.zeros(3),  # TODO: センサから取得する
            timestamp=float(d.time),
        )

    def send_joint_positions(self, positions: np.ndarray) -> None:
        self._data.ctrl[:] = self._validate(positions)
        for _ in range(self._n_substeps):
            self._mujoco.mj_step(self._model, self._data)

    def close(self) -> None:
        pass
