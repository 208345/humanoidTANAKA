"""学習済みモデルの評価・可視化。

    python -m learning.eval.evaluate --urdf model/humanoid.urdf --model learning/policies/latest/final.zip

学習済みポリシーを GUI 上で動作確認する。
"""

from __future__ import annotations
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.envs.pybullet_env import PyBulletHumanoidEnv
from learning.envs.unity_env import UnityHumanoidEnv
from learning.train.rewards import StandingReward


def main() -> None:
    parser = argparse.ArgumentParser(description="学習済みモデルの評価")
    parser.add_argument("--urdf", type=str, required=True, help="URDF ファイルパス")
    parser.add_argument("--model", type=str, required=True, help="モデルファイルパス (.zip)")
    parser.add_argument("--vecnorm", type=str, default=None, help="VecNormalize 統計 (.pkl)")
    parser.add_argument("--params", type=str, default="model/params.yaml")

    parser.add_argument("--episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument(
        "--backend", type=str, default="pybullet", choices=["pybullet", "unity"],
        help="シミュレーションバックエンド (pybullet | unity)"
    )
    parser.add_argument(
        "--env-path", type=str, default=None,
        help="Unityバックエンド使用時のビルド済み実行ファイル (.exe) のパス"
    )

    args = parser.parse_args()

    print("=== ヒューマノイド強化学習 評価スクリプト ===")
    print(f"  モデル: {args.model}")

    # 環境の構築（GUI 表示）
    from learning.envs.base_env import load_robot_params
    params = load_robot_params(args.params)
    num_joints = params['robot']['num_joints']
    reward_fn = StandingReward(num_joints=num_joints)

    if args.backend == "unity":
        env = UnityHumanoidEnv(
            urdf_path=args.urdf,
            params_path=args.params,
            reward_fn=reward_fn,
            render_mode="human",
            file_name=args.env_path
        )
    else:
        env = PyBulletHumanoidEnv(
            urdf_path=args.urdf,
            params_path=args.params,
            reward_fn=reward_fn,
            render_mode="human",
        )

    vec_env = DummyVecEnv([lambda: env])

    # VecNormalize の復元
    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False  # 評価時は統計を更新しない
        vec_env.norm_reward = False

    # モデルの読み込み
    model = PPO.load(args.model)

    # 評価ループ
    total_rewards = []
    total_steps_list = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        episode_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            steps += 1

            time.sleep(1.0 / 50.0)  # 実時間表示

            if done[0]:
                is_success = info[0].get("is_timeout_success", False)
                status = "✅ 成功（時間切れ）" if is_success else "❌ 転倒"
                print(f"  Episode {ep+1}: {status}  報酬={episode_reward:.1f}  ステップ={steps}")
                total_rewards.append(episode_reward)
                total_steps_list.append(steps)
                time.sleep(2.0)
                break

    # 統計の表示
    import numpy as np
    rewards = np.array(total_rewards)
    steps_arr = np.array(total_steps_list)
    print(f"\n--- 評価結果 ({args.episodes} エピソード) ---")
    print(f"  平均報酬: {rewards.mean():.1f} ± {rewards.std():.1f}")
    print(f"  平均ステップ: {steps_arr.mean():.0f} ± {steps_arr.std():.0f}")
    print(f"  成功率: {(steps_arr >= 1000).sum()}/{args.episodes}")

    print('\nすべてのエピソードが終了しました。5秒後にウィンドウを閉じます...')
    time.sleep(5.0)
    vec_env.close()


if __name__ == "__main__":
    main()


