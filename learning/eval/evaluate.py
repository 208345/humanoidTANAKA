from __future__ import annotations
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse
import time
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.envs.base_env import load_robot_params
from learning.envs.pybullet_env import PyBulletHumanoidEnv
from learning.train.rewards import StandingReward, WalkingReward


def main() -> None:
    parser = argparse.ArgumentParser(description="学習済みモデルの評価")
    parser.add_argument("--urdf", type=str, required=True, help="URDF ファイルパス")
    parser.add_argument("--model", type=str, required=True, help="モデルファイルパス (.zip)")
    parser.add_argument("--vecnorm", type=str, default=None, help="VecNormalize 統計 (.pkl)")
    parser.add_argument("--params", type=str, default="model/params.yaml")
    parser.add_argument("--curriculum", type=str, default=None, help="カリキュラム設定 (.yaml)")
    parser.add_argument("--reward-type", type=str, default=None, choices=["standing", "walking"])
    parser.add_argument("--episodes", type=int, default=5, help="評価エピソード数")
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
    print(f"  パラメータ: {args.params}")

    params = load_robot_params(args.params)
    num_joints = params['robot']['num_joints']

    # 報酬関数の設定: params_residual や curriculum があれば WalkingReward を適用
    is_residual = "residual" in str(args.params) or "res1" in str(args.model)
    curriculum_path = args.curriculum
    if curriculum_path is None and is_residual:
        curriculum_path = "learning/train/curriculum_residual.yaml"

    reward_params = {}
    if curriculum_path and os.path.exists(curriculum_path):
        with open(curriculum_path, encoding="utf-8") as f:
            reward_params = yaml.safe_load(f)["phases"][-1].get("reward_params", {})

    reward_type = args.reward_type
    if reward_type is None:
        reward_type = "walking" if (is_residual or "walk" in str(args.model)) else "standing"

    if reward_type == "walking":
        reward_fn = WalkingReward(num_joints=num_joints, **reward_params)
    else:
        reward_fn = StandingReward(num_joints=num_joints)

    if args.backend == "unity":
        from learning.envs.unity_env import UnityHumanoidEnv
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
        vec_env.training = False
        vec_env.norm_reward = False

    # モデルの読み込み (Python 3.11 と 3.13 のバイトコード互換性クラッシュを防ぐため custom_objects を指定)
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.2,
        "clip_range_vf": None,
    }
    model = PPO.load(args.model, env=vec_env, device="cpu", custom_objects=custom_objects)
    print("モデルの読み込みに成功しました。シミュレーションを開始します！")

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

            time.sleep(1.0 / 50.0)  # 実時間表示 (50Hz)

            if done[0]:
                is_success = info[0].get("is_timeout_success", False)
                status = "✅ 成功（時間切れ）" if is_success else "❌ 転倒"
                print(f"  Episode {ep+1}: {status}  報酬={episode_reward:.1f}  ステップ={steps}")
                total_rewards.append(episode_reward)
                total_steps_list.append(steps)
                time.sleep(1.5)
                break

    rewards = np.array(total_rewards)
    steps_arr = np.array(total_steps_list)
    print(f"\n--- 評価結果 ({args.episodes} エピソード) ---")
    print(f"  平均報酬: {rewards.mean():.1f} ± {rewards.std():.1f}")
    print(f"  平均ステップ: {steps_arr.mean():.0f} ± {steps_arr.std():.0f}")

    print('\n全エピソードが終了しました。5秒後にウィンドウを閉じます...')
    time.sleep(5.0)
    vec_env.close()


if __name__ == "__main__":
    main()
