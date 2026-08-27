import sys

with open('learning/eval/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add imports
content = content.replace(
    "from learning.envs.pybullet_env import PyBulletHumanoidEnv",
    "from learning.envs.pybullet_env import PyBulletHumanoidEnv\nfrom learning.envs.unity_env import UnityHumanoidEnv"
)

# Add arguments
arg_str = """
    parser.add_argument("--episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument(
        "--backend", type=str, default="pybullet", choices=["pybullet", "unity"],
        help="シミュレーションバックエンド (pybullet | unity)"
    )
    parser.add_argument(
        "--env-path", type=str, default=None,
        help="Unityバックエンド使用時のビルド済み実行ファイル (.exe) のパス"
    )
"""
content = content.replace("    parser.add_argument(\"--episodes\", type=int, default=10, help=\"評価エピソード数\")", arg_str)

# Update environment creation
env_str = """
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
"""
content = content.replace(
"""    env = PyBulletHumanoidEnv(
        urdf_path=args.urdf,
        params_path=args.params,
        reward_fn=reward_fn,
        render_mode="human",
    )""",
env_str
)

with open('learning/eval/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("evaluate.py updated.")
