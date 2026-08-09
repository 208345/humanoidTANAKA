"""制御ループの実行エントリポイント。

    python -m control.run --backend dummy --controller zero --steps 100
    python -m control.run --backend sim   --controller zero --steps 1000
    python -m control.run --backend real  --controller zero --steps 500

バックエンドと制御器を入れ替えても、このループ自体は一切変わらない。
それがこの設計の目的。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from control.interface import make_controller, make_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="制御ループを実行する")
    parser.add_argument(
        "--backend", default="dummy", choices=["dummy", "sim", "real"],
        help="ロボットの実体",
    )
    parser.add_argument(
        "--controller", default="zero", choices=["zero", "policy"],
        help="制御器",
    )
    parser.add_argument("--steps", type=int, default=100, help="実行ステップ数")
    parser.add_argument(
        "--realtime", action="store_true",
        help="実時間に合わせて待つ（sim を目視確認するとき用）",
    )
    args = parser.parse_args()

    with make_robot(args.backend) as robot:
        controller = make_controller(args.controller, robot.num_joints)
        controller.reset()
        state = robot.reset()

        print(f"backend={args.backend} controller={args.controller} "
              f"joints={robot.num_joints} dt={robot.control_dt}s")

        for i in range(args.steps):
            loop_start = time.perf_counter()

            command = controller(state)
            robot.send_joint_positions(command)
            state = robot.read_state()

            if i % 50 == 0:
                print(f"[{i:5d}] t={state.timestamp:7.3f}s "
                      f"|q|={np.linalg.norm(state.joint_positions):.4f}")

            if args.realtime:
                elapsed = time.perf_counter() - loop_start
                # TODO: 遅れが常態化していないか監視すること。
                #       実機では周期が守れないこと自体が不具合。
                time.sleep(max(robot.control_dt - elapsed, 0.0))

    print("完了")


if __name__ == "__main__":
    main()
