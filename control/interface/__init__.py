"""sim / 実機 / 学習ポリシーを差し替え可能にする境界層。

このパッケージの抽象クラスを変更するときは、必ず3人で合意すること。
ここが全員の作業をつなぐ唯一の接点。
"""

from control.interface.controller import Controller, PolicyController, ZeroController
from control.interface.factory import make_controller, make_robot
from control.interface.robot_interface import RobotInterface, RobotState

__all__ = [
    "RobotInterface",
    "RobotState",
    "Controller",
    "ZeroController",
    "PolicyController",
    "make_robot",
    "make_controller",
]
