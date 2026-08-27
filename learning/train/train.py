"""学習のエントリポイント。

    python -m learning.train.train --urdf model/humanoid.urdf
    python -m learning.train.train --urdf model/humanoid.urdf --resume learning/policies/latest/100k.zip

cube-sim-rl の改善点をすべて組み込み済み:
    - VecNormalize（観測値・報酬の自動正規化）
    - SubprocVecEnv（並列環境による学習高速化）
    - 学習率の線形減衰
    - カリキュラム学習の自動管理
    - チェックポイントの自動保存
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from learning.envs.pybullet_env import PyBulletHumanoidEnv
from learning.train.callbacks import CheckpointCallback, CurriculumCallback
from learning.train.curriculum import CurriculumManager
from learning.train.rewards import StandingReward, WalkingReward


def linear_schedule(initial_value: float):
    """学習率を線形減衰させるスケジューラ。

    学習終盤での方策崩壊・破局的忘却を防止する。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(
    urdf_path: str,
    params_path: str,
    reward_fn,
    rank: int,
    seed: int = 42,
):
    """並列環境の1つを生成するファクトリ関数。"""
    def _init():
        env = PyBulletHumanoidEnv(
            urdf_path=urdf_path,
            params_path=params_path,
            reward_fn=reward_fn,
            render_mode=None,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_reward_fn(reward_type: str, num_joints: int, **kwargs):
    """報酬関数を生成する。"""
    if reward_type == "standing":
        return StandingReward(num_joints=num_joints, **kwargs)
    elif reward_type == "walking":
        return WalkingReward(num_joints=num_joints, **kwargs)
    else:
        raise ValueError(f"未知の報酬タイプ: {reward_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ヒューマノイド強化学習 (PPO) 学習スクリプト"
    )
    parser.add_argument(
        "--urdf", type=str, required=True,
        help="ロボットの URDF ファイルパス",
    )
    parser.add_argument(
        "--params", type=str, default="model/params.yaml",
        help="ロボットパラメータファイル",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="チェックポイントから再開する場合のパス (.zip)",
    )
    parser.add_argument(
        "--resume-vecnorm", type=str, default=None,
        help="VecNormalize の統計ファイルパス (.pkl)",
    )
    parser.add_argument(
        "--curriculum", type=str, default=None,
        help="カリキュラム設定ファイルパス (.yaml)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="総学習ステップ数",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4,
        help="並列環境数（CPU コア数以下を推奨）",
    )
    parser.add_argument(
        "--save-path", type=str, default="learning/policies/latest",
        help="チェックポイントの保存先",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--reward-type", type=str, default="standing",
        choices=["standing", "walking"],
        help="報酬関数の種類（カリキュラム未使用時）",
    )
    args = parser.parse_args()

    print("=== ヒューマノイド強化学習 (PPO) 学習スクリプト ===")
    print(f"  URDF: {args.urdf}")
    print(f"  並列環境数: {args.num_envs}")
    print(f"  総ステップ数: {args.total_timesteps:,}")

    # 乱数シード
    set_random_seed(args.seed)

    # 報酬関数（カリキュラム使用時は後で上書きされる）
    # num_joints は params.yaml から取得
    import yaml
    with open(args.params, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    num_joints = params["robot"]["num_joints"]

    reward_fn = make_reward_fn(args.reward_type, num_joints=num_joints)

    # --- 環境の構築 ---
    if args.num_envs == 1:
        vec_env = DummyVecEnv([make_env(args.urdf, args.params, reward_fn, 0, args.seed)])
    else:
        vec_env = SubprocVecEnv([
            make_env(args.urdf, args.params, reward_fn, i, args.seed)
            for i in range(args.num_envs)
        ])

    # VecNormalize で観測値・報酬を自動正規化
    if args.resume_vecnorm:
        vec_env = VecNormalize.load(args.resume_vecnorm, vec_env)
        print(f"  VecNormalize 復元: {args.resume_vecnorm}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- PPO ハイエンドパラメータ（ロボティクス制御向け）---
    ppo_kwargs = {
        "learning_rate": linear_schedule(1e-4),
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.995,
        "ent_coef": 0.005,
    }
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # --- モデルの構築 ---
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"\n【Resume】{args.resume} から学習を再開します。")
        model = PPO.load(
            args.resume, env=vec_env,
            tensorboard_log=str(save_path / "logs"),
            custom_objects={
                "learning_rate": ppo_kwargs["learning_rate"],
                "n_steps": ppo_kwargs["n_steps"],
                "batch_size": ppo_kwargs["batch_size"],
                "gamma": ppo_kwargs["gamma"],
                "ent_coef": ppo_kwargs["ent_coef"],
            },
        )
        reset_timesteps = False
    else:
        print("\n【Initialize】新規モデルの学習を開始します。")
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            tensorboard_log=str(save_path / "logs"),
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )
        reset_timesteps = True

    # --- コールバックの設定 ---
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=str(save_path)),
    ]

    # カリキュラム学習
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumManager(args.curriculum)
        phase = curriculum.current_phase
        print(f"  カリキュラム: {phase.name}")

        # 最初のフェーズの報酬関数を適用
        reward_config = curriculum.get_reward_params()
        initial_reward = make_reward_fn(
            reward_config["type"], num_joints=num_joints, **reward_config["params"]
        )
        # SubprocVecEnv の場合は各環境への適用が必要
        # → 初期生成時に reward_fn を渡しているので、カリキュラムコールバックで更新

        callbacks.append(CurriculumCallback(
            curriculum=curriculum,
            env=vec_env,
        ))

    callback_list = CallbackList(callbacks)

    # --- 学習実行 ---
    print(f"\n学習を開始します... (目標: {args.total_timesteps:,} ステップ)")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        reset_num_timesteps=reset_timesteps,
    )

    # 最終モデルの保存
    final_path = save_path / "final.zip"
    model.save(str(final_path))
    vec_env.save(str(save_path / "final_vecnorm.pkl"))
    print(f"\n学習完了。最終モデル: {final_path}")


if __name__ == "__main__":
    main()