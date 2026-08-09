"""物理シミュレーション無しで動く偽物のロボット。

MuJoCo も実機も無い状態でインターフェースとテストを回すために使う。
「送った指令が一次遅れで反映される」だけの、極めて単純なモデル。

CI はこれで回すので、常に依存ライブラリ無しで動くこと。
"""

from __future__ import annotations

import numpy as np

from control.interface.robot_interface import RobotInterface, RobotState


class DummyRobot(RobotInterface):
    def __init__(self, num_joints: int = 12, control_dt: float = 0.02) -> None:
        self._num_joints = num_joints
        self._control_dt = control_dt
        self._q = np.zeros(num_joints)
        self._dq = np.zeros(num_joints)
        self._target = np.zeros(num_joints)
        self._t = 0.0
        # 一次遅れの時定数 [s]。実機サーボの追従遅れを雑に模したもの。
        self._tau = 0.05

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def reset(self) -> RobotState:
        self._q[:] = 0.0
        self._dq[:] = 0.0
        self._target[:] = 0.0
        self._t = 0.0
        return self.read_state()

    def read_state(self) -> RobotState:
        return RobotState(
            joint_positions=self._q.copy(),
            joint_velocities=self._dq.copy(),
            imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            imu_angular_velocity=np.zeros(3),
            imu_linear_acceleration=np.array([0.0, 0.0, 9.81]),
            timestamp=self._t,
        )

    def send_joint_positions(self, positions: np.ndarray) -> None:
        self._target = self._validate(positions)
        # 一次遅れ: q += (target - q) * dt / tau
        alpha = min(self._control_dt / self._tau, 1.0)
        new_q = self._q + (self._target - self._q) * alpha
        self._dq = (new_q - self._q) / self._control_dt
        self._q = new_q
        self._t += self._control_dt

    def close(self) -> None:
        pass
