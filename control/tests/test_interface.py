"""インターフェース層のテスト。

ここが壊れると3人全員の作業が壊れるので、CI で常に検証する。
"""

import numpy as np
import pytest

from control.interface import make_controller, make_robot
from control.interface.dummy_robot import DummyRobot
from control.interface.robot_interface import RobotState


def test_dummy_robot_reset_returns_zero_state():
    robot = DummyRobot(num_joints=12)
    state = robot.reset()
    assert isinstance(state, RobotState)
    assert state.joint_positions.shape == (12,)
    np.testing.assert_allclose(state.joint_positions, 0.0)


def test_joint_command_moves_toward_target():
    """指令を送り続ければ目標角に近づくこと。"""
    robot = DummyRobot(num_joints=12)
    robot.reset()
    target = np.full(12, 0.5)
    for _ in range(500):
        robot.send_joint_positions(target)
    state = robot.read_state()
    np.testing.assert_allclose(state.joint_positions, target, atol=1e-3)


def test_wrong_shape_is_rejected():
    """形の違う指令は必ず弾く。実機に流れると事故になる。"""
    robot = DummyRobot(num_joints=12)
    with pytest.raises(ValueError):
        robot.send_joint_positions(np.zeros(6))


def test_nan_command_is_rejected():
    """NaN をサーボに送ると実機が暴れる。境界で止める。"""
    robot = DummyRobot(num_joints=12)
    bad = np.zeros(12)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        robot.send_joint_positions(bad)


def test_context_manager_closes():
    with DummyRobot() as robot:
        robot.reset()


def test_factory_creates_dummy():
    robot = make_robot("dummy")
    assert robot.num_joints == 12


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_robot("nonexistent")


def test_zero_controller_returns_correct_shape():
    controller = make_controller("zero", num_joints=12)
    controller.reset()
    state = DummyRobot(num_joints=12).reset()
    command = controller(state)
    assert command.shape == (12,)
    np.testing.assert_allclose(command, 0.0)


def test_full_loop_runs():
    """バックエンドと制御器を組み合わせた制御ループが一通り回ること。"""
    with make_robot("dummy") as robot:
        controller = make_controller("zero", robot.num_joints)
        controller.reset()
        state = robot.reset()
        for _ in range(100):
            robot.send_joint_positions(controller(state))
            state = robot.read_state()
        assert state.timestamp > 0.0
