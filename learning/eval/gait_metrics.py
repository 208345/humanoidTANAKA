"""学習済みポリシーの歩容を GUI なしで数値評価する。

    python -m learning.eval.gait_metrics --urdf model/humanoid.urdf \
        --model ~/humanoid_runs/walk2/final.zip --vecnorm ~/humanoid_runs/walk2/final_vecnorm.pkl

ep_len / ep_rew だけでは「その場で立っている」と「歩いている」の区別が
つかないので、前進距離・横ドリフト・足の交互接地などを実測する。
"""

from __future__ import annotations
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from learning.envs.base_env import load_robot_params
from learning.envs.pybullet_env import PyBulletHumanoidEnv
from learning.train.rewards import WalkingReward


def main() -> None:
    parser = argparse.ArgumentParser(description="歩容の数値評価")
    parser.add_argument("--urdf", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--params", type=str, default="model/params.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--curriculum", type=str, default="learning/train/curriculum_config.yaml",
                        help="最終フェーズの reward_params を評価にも使う（終了条件を学習と揃える）")
    args = parser.parse_args()

    params = load_robot_params(args.params)
    num_joints = params['robot']['num_joints']
    with open(args.curriculum, encoding="utf-8") as f:
        reward_params = yaml.safe_load(f)["phases"][-1]["reward_params"]
    env = PyBulletHumanoidEnv(
        urdf_path=args.urdf, params_path=args.params,
        reward_fn=WalkingReward(num_joints=num_joints, **reward_params),
    )
    vec_env = DummyVecEnv([lambda: env])
    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(args.model, device="cpu")
    p = env._p

    rows = []
    for ep in range(args.episodes):
        obs = vec_env.reset()
        start = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])
        steps, contacts, heights, yaws, phase_match = 0, [], [], [], []
        lift = [[], []]   # 足ごと: 自分の遊脚期に地面から離れていたか
        max_clear = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = vec_env.step(action)
            steps += 1
            if done[0]:
                fell = not info[0].get("TimeLimit.truncated", False)
                break
            st = env._read_state()
            contacts.append(st["foot_contacts"].copy())
            heights.append(st["base_height"])
            yaws.append(info[0].get("yaw", 0.0))
            # 位相どおりの接地か（φ<0.5 は足0が遊脚）。報酬の gait_weight 項と同じ定義
            u = (env._gait_phase % 0.5) / 0.5
            if abs(u - 0.5) < reward_params.get("swing_ratio", 1.0) / 2.0:
                swing = np.array([1.0, 0.0]) if env._gait_phase < 0.5 else np.array([0.0, 1.0])
            else:
                swing = np.zeros(2)
            fc = st["foot_contacts"]
            phase_match.append(float(np.mean(fc * (1.0 - swing) + (1.0 - fc) * swing)))
            for k in range(2):
                if swing[k] > 0.5:
                    lift[k].append(1.0 - fc[k])
            max_clear = max(max_clear, float(np.max(st["foot_heights"])) - 0.014)
            end = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])

        t = steps * env.control_dt
        c = np.array(contacts) if contacts else np.zeros((1, 2))
        single = float(np.mean(c.sum(axis=1) == 1))
        both = float(np.mean(c.sum(axis=1) == 2))
        air = float(np.mean(c.sum(axis=1) == 0))
        # 接地の切り替わり回数（左右それぞれ 0→1 の回数 = 着地回数）
        touchdowns = int(np.sum(np.diff(c, axis=0) > 0.5))
        d = end - start
        rows.append(dict(
            steps=steps, fell=fell, time=t,
            fwd=d[1], lat=d[0], v=d[1] / t if t > 0 else 0.0,
            single=single, both=both, air=air, touchdowns=touchdowns,
            steps_per_s=touchdowns / t if t > 0 else 0.0,
            height=float(np.mean(heights)) if heights else 0.0,
            yaw_end=float(np.degrees(yaws[-1])) if yaws else 0.0,
            clear=max_clear,
            match=float(np.mean(phase_match)) if phase_match else 0.0,
            liftL=float(np.mean(lift[0])) if lift[0] else 0.0,
            liftR=float(np.mean(lift[1])) if lift[1] else 0.0,
        ))
        r = rows[-1]
        print(f"Ep{ep+1}: {'転倒' if fell else '完走'} {steps}step ({t:.1f}s) | "
              f"前進 {r['fwd']:+.3f}m 横 {r['lat']:+.3f}m 平均 {r['v']:+.3f}m/s | "
              f"片足 {single:.0%} 両足 {both:.0%} 空中 {air:.0%} | "
              f"着地 {touchdowns}回 ({r['steps_per_s']:.1f}/s) 位相一致 {r['match']:.0%} | "
              f"胴高 {r['height']:.3f} yaw {r['yaw_end']:+.0f}° 足上げ最大 {max_clear*1000:.0f}mm")

    m = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != "fell"}
    print(f"\n平均: 前進速度 {m['v']:+.3f} m/s (目標 0.2), 横ドリフト {m['lat']:+.3f} m, "
          f"片足支持 {m['single']:.0%}, 位相一致 {m['match']:.0%}, 着地 {m['steps_per_s']:.1f}回/s, "
          f"足上げ率 L{m['liftL']:.0%}/R{m['liftR']:.0%}, 胴高 {m['height']:.3f}{' ⚠座り込み' if m['height'] < 0.12 else ''}, "
          f"転倒 {sum(r['fell'] for r in rows)}/{len(rows)}")
    vec_env.close()


if __name__ == "__main__":
    main()
