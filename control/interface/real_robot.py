"""実機との通信。

通信フォーマットは docs/interfaces/protocol.md が正。
このファイルと protocol.md がずれたら、必ず protocol.md を先に直す。

Phase 2 の課題: このクラスを完成させ、Phase 1 と同じ Controller で
実機の全関節が動くことを確認する。ここがプロジェクト最初の関門。
"""

from __future__ import annotations

import numpy as np

from control.interface.robot_interface import RobotInterface, RobotState


class RealRobot(RobotInterface):
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 921600,
        num_joints: int = 12,
        control_dt: float = 0.02,
    ) -> None:
        try:
            import serial  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyserial が必要です。'pip install -e \".[real]\"' を実行してください。"
            ) from exc

        import serial

        self._serial = serial.Serial(port, baudrate, timeout=0.05)
        self._num_joints = num_joints
        self._control_dt = control_dt

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def reset(self) -> RobotState:
        # 安全上の注意: 実機ではいきなり初期姿勢へ飛ばさないこと。
        # 数秒かけて現在角から初期姿勢へ補間して移動させる実装にする。
        raise NotImplementedError("Phase 2 で実装する（ゆっくり初期姿勢へ移行させる）")

    def read_state(self) -> RobotState:
        raise NotImplementedError("Phase 2 で実装する（protocol.md の受信パケットを解釈）")

    def send_joint_positions(self, positions: np.ndarray) -> None:
        positions = self._validate(positions)
        raise NotImplementedError("Phase 2 で実装する（protocol.md の送信パケットを組む）")

    def close(self) -> None:
        # 脱力させてからポートを閉じる。閉じ忘れるとサーボが最後の指令を保持し続けて焼ける。
        self._serial.close()
