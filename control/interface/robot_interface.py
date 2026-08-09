"""ロボット本体との境界。

制御コードは「相手が sim か実機か」を知らない。この約束がプロジェクトの土台。

単位はすべて SI で統一する:
    角度 [rad] / 角速度 [rad/s] / 長さ [m] / 時間 [s]
符号と原点姿勢の定義は docs/interfaces/joints.md を参照。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobotState:
    """ある瞬間のロボットの観測値。

    Attributes:
        joint_positions: 関節角 [rad]。並び順は model/params.yaml の joints に従う。
        joint_velocities: 関節角速度 [rad/s]。
        imu_quaternion: 胴体の姿勢 (w, x, y, z)。ワールド座標から見た胴体の向き。
        imu_angular_velocity: 胴体の角速度 [rad/s]。胴体座標系で表現。
        imu_linear_acceleration: 胴体の加速度 [m/s^2]。重力を含む。胴体座標系。
        timestamp: 観測時刻 [s]。sim ではシミュレーション時刻、実機では受信時刻。
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    imu_quaternion: np.ndarray
    imu_angular_velocity: np.ndarray
    imu_linear_acceleration: np.ndarray
    timestamp: float
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 型のゆらぎ（list が渡される等）をここで吸収しておく。
        # 下流で「なぜか float64 じゃない」というバグに悩まされないため。
        self.joint_positions = np.asarray(self.joint_positions, dtype=np.float64)
        self.joint_velocities = np.asarray(self.joint_velocities, dtype=np.float64)
        self.imu_quaternion = np.asarray(self.imu_quaternion, dtype=np.float64)
        self.imu_angular_velocity = np.asarray(
            self.imu_angular_velocity, dtype=np.float64
        )
        self.imu_linear_acceleration = np.asarray(
            self.imu_linear_acceleration, dtype=np.float64
        )


class RobotInterface(ABC):
    """ロボット実体の共通インターフェース。

    SimRobot / RealRobot / DummyRobot がこれを実装する。
    制御側はこの型だけを見る。
    """

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """関節数。"""

    @property
    @abstractmethod
    def control_dt(self) -> float:
        """制御周期 [s]。ポリシーの推論周期と一致させる。"""

    @abstractmethod
    def reset(self) -> RobotState:
        """初期姿勢に戻し、最初の観測を返す。

        実機では「安全な姿勢にゆっくり移動する」処理になる。
        いきなり初期姿勢へ飛ばすと機体が壊れるので注意。
        """

    @abstractmethod
    def read_state(self) -> RobotState:
        """現在の観測を取得する。"""

    @abstractmethod
    def send_joint_positions(self, positions: np.ndarray) -> None:
        """関節角の目標値を送る [rad]。

        Raises:
            ValueError: 長さが num_joints と一致しない場合。
        """

    @abstractmethod
    def close(self) -> None:
        """後始末（シリアルポートを閉じる、サーボを脱力させる等）。"""

    # --- 以下は共通実装 ---

    def _validate(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.num_joints,):
            raise ValueError(
                f"関節指令の形が不正です: {positions.shape}, "
                f"期待値: ({self.num_joints},)"
            )
        if not np.all(np.isfinite(positions)):
            # NaN をサーボに送ると実機が暴れる。ここで必ず止める。
            raise ValueError("関節指令に NaN または inf が含まれています")
        return positions

    def __enter__(self) -> RobotInterface:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
