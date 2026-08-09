"""起動引数から実体を組み立てる。

「どのバックエンドを使うか」の分岐をこのファイル1箇所に閉じ込める。
制御ロジック側に if backend == "sim" のような分岐を書かないこと。
"""

from __future__ import annotations

from control.interface.controller import Controller, PolicyController, ZeroController
from control.interface.robot_interface import RobotInterface


def make_robot(backend: str, **kwargs: object) -> RobotInterface:
    if backend == "dummy":
        from control.interface.dummy_robot import DummyRobot

        return DummyRobot(**kwargs)  # type: ignore[arg-type]
    if backend == "sim":
        from control.interface.sim_robot import SimRobot

        return SimRobot(**kwargs)  # type: ignore[arg-type]
    if backend == "real":
        from control.interface.real_robot import RealRobot

        return RealRobot(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"未知のバックエンドです: {backend!r} (dummy / sim / real)")


def make_controller(name: str, num_joints: int, **kwargs: object) -> Controller:
    if name == "zero":
        return ZeroController(num_joints)
    if name == "policy":
        return PolicyController(num_joints=num_joints, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"未知の制御器です: {name!r} (zero / policy)")
