"""制御器の共通インターフェース。

手で設計した制御器も、学習した強化学習ポリシーも、
「観測を受け取って関節指令を返すもの」という同じ型にする。

こうしておくと、両者を同じ実験スクリプトで比較できる。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from control.interface.robot_interface import RobotState


class Controller(ABC):
    """観測 → 関節指令 の写像。"""

    @abstractmethod
    def reset(self) -> None:
        """エピソード開始時の初期化（内部状態を持つ制御器のため）。"""

    @abstractmethod
    def __call__(self, state: RobotState) -> np.ndarray:
        """関節角の目標値を返す [rad]。"""


class ZeroController(Controller):
    """全関節を 0 [rad]（原点姿勢）に保つだけの制御器。

    Phase 1 / Phase 2 の疎通確認用。
    「sim と実機が同じコードで動く」ことを最初に確認するために使う。
    """

    def __init__(self, num_joints: int) -> None:
        self.num_joints = num_joints

    def reset(self) -> None:
        pass

    def __call__(self, state: RobotState) -> np.ndarray:
        return np.zeros(self.num_joints, dtype=np.float64)


class PolicyController(Controller):
    """学習済みの強化学習ポリシーを動かす制御器。

    Phase 4 以降で使う。今は骨組みだけ。

    実装時の注意:
        - 観測の作り方（正規化・履歴の積み方）を学習時と完全に一致させること。
          ここがずれるのが sim-to-real 失敗の最頻出原因。
          観測の仕様は learning/envs/ 側と共有し、二重に書かない。
        - 出力は多くの場合「初期姿勢からの差分」なので、
          そのまま絶対角として送らないこと。
    """

    def __init__(self, policy_path: str | Path, num_joints: int) -> None:
        self.policy_path = Path(policy_path)
        self.num_joints = num_joints
        self._policy = None

    def load(self) -> None:
        raise NotImplementedError(
            "Phase 4 で実装する。学習済みポリシーの読み込みをここに書く。"
        )

    def reset(self) -> None:
        pass

    def __call__(self, state: RobotState) -> np.ndarray:
        raise NotImplementedError("Phase 4 で実装する。")
