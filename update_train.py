import sys

with open('learning/train/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add imports
content = content.replace(
    "from learning.envs.pybullet_env import PyBulletHumanoidEnv",
    "from learning.envs.pybullet_env import PyBulletHumanoidEnv\nfrom learning.envs.unity_env import UnityHumanoidEnv"
)

# Add arguments
arg_str = """
    parser.add_argument(
        "--backend", type=str, default="pybullet", choices=["pybullet", "unity"],
        help="シミュレーションバックエンド (pybullet | unity)"
    )
    parser.add_argument(
        "--env-path", type=str, default=None,
        help="Unityバックエンド使用時のビルド済み実行ファイル (.exe) のパス"
    )
    parser.add_argument(
"""
content = content.replace("    parser.add_argument(", arg_str, 1)

# Update make_env definition to accept backend and env_path
content = content.replace(
    "def make_env(\n    urdf_path: str,\n    params_path: str,\n    reward_fn,\n    rank: int,\n    seed: int = 42,\n):",
    "def make_env(\n    urdf_path: str,\n    params_path: str,\n    reward_fn,\n    rank: int,\n    seed: int = 42,\n    backend: str = 'pybullet',\n    env_path: str = None,\n):"
)

# Update _init inside make_env
env_init_str = """
        if backend == "unity":
            # Workerごとのポート衝突を防ぐため、Unity側で Worker ID を割り当てる機能などは今回は簡略化
            # env_path が指定されていればそれを使う
            env = UnityHumanoidEnv(
                urdf_path=urdf_path,
                params_path=params_path,
                reward_fn=reward_fn,
                render_mode=None,
                file_name=env_path
            )
        else:
            env = PyBulletHumanoidEnv(
                urdf_path=urdf_path,
                params_path=params_path,
                reward_fn=reward_fn,
                render_mode=None,
            )
"""
content = content.replace(
"""        env = PyBulletHumanoidEnv(
            urdf_path=urdf_path,
            params_path=params_path,
            reward_fn=reward_fn,
            render_mode=None,
        )""",
env_init_str
)

# Update make_env calls
content = content.replace(
    "make_env(args.urdf, args.params, reward_fn, 0, args.seed)",
    "make_env(args.urdf, args.params, reward_fn, 0, args.seed, args.backend, args.env_path)"
)
content = content.replace(
    "make_env(args.urdf, args.params, reward_fn, i, args.seed)",
    "make_env(args.urdf, args.params, reward_fn, i, args.seed, args.backend, args.env_path)"
)

with open('learning/train/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("train.py updated.")
