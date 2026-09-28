"""学習済みポリシーの動きを MP4 に書き出す（GUI 不要・画面なし描画）。

    python -m learning.eval.record_video <checkpoint(拡張子なし)> <out.mp4> [seconds] [noterm] [--residual]
左: 横から / 右: 正面から。足の接地を L/R の丸で表示。
"""
import sys
import numpy as np
import imageio
import yaml
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from learning.envs.pybullet_env import PyBulletHumanoidEnv
from learning.train.rewards import WalkingReward

ckpt, out = sys.argv[1], sys.argv[2]
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
noterm = len(sys.argv) > 4 and sys.argv[4] == "noterm"   # 座り込み判定を外す（旧ポリシー比較用）
W, H, FPS = 480, 360, 25

PARAMS = "model/params_residual.yaml" if "--residual" in sys.argv else "model/params.yaml"
CURR = "learning/train/curriculum_residual.yaml" if "--residual" in sys.argv else "learning/train/curriculum_config.yaml"
with open(CURR, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["phases"][-1]["reward_params"]
if noterm:
    cfg = {k: v for k, v in cfg.items() if k not in ("min_height", "terminate_on_body_contact")}
env = PyBulletHumanoidEnv("model/humanoid.urdf", params_path=PARAMS, reward_fn=WalkingReward(num_joints=8, **cfg))
ve = VecNormalize.load(ckpt + "_vecnorm.pkl", DummyVecEnv([lambda: env]))
ve.training = False
ve.norm_reward = False
model = PPO.load(ckpt + ".zip", device="cpu")
p = env._p

proj = p.computeProjectionMatrixFOV(fov=45, aspect=W / H, nearVal=0.01, farVal=5)


def shot(target, yaw):
    view = p.computeViewMatrixFromYawPitchRoll(target, 0.55, yaw, -15, 0, 2)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    return np.reshape(np.asarray(img, dtype=np.uint8), (H, W, 4))[:, :, :3]


writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7)
obs = ve.reset()
start = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])
last = start.copy()
n_steps = int(seconds * env.control_hz)
ep, ep_t, falls = 1, 0, 0
for t in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = ve.step(action)
    ep_t += 1
    if done[0]:
        fell = not info[0].get("TimeLimit.truncated", False)
        falls += fell
        print(f"episode {ep} {'fell' if fell else 'timeout'} at {ep_t / env.control_hz:.1f}s, "
              f"forward {last[1] - start[1]:+.2f} m", flush=True)
        # 自動リセット済み。次のエピソードも続けて撮る
        start = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])
        ep, ep_t = ep + 1, 0
    last = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])
    if t % (env.control_hz // FPS) != 0:
        continue
    pos = last
    target = [pos[0], pos[1], 0.08]
    frame = Image.fromarray(np.concatenate([shot(target, 90), shot(target, 180)], axis=1))
    d = ImageDraw.Draw(frame)
    st = env._read_state()
    d.text((10, 8), f"{ckpt.split('/')[-2]}/{ckpt.split('/')[-1]}  ep{ep} t={ep_t / env.control_hz:4.1f}s  "
                    f"forward {pos[1] - start[1]:+.2f} m  height {st['base_height']:.3f} m",
           fill=(0, 0, 0))
    for k, name in enumerate(["L", "R"]):
        on = st["foot_contacts"][k] > 0.5
        x = 10 + k * 40
        d.ellipse([x, 26, x + 16, 42], fill=(40, 160, 60) if on else (230, 230, 230), outline=(0, 0, 0))
        d.text((x + 20, 28), name, fill=(0, 0, 0))
    d.text((W + 10, 8), "front view", fill=(0, 0, 0))
    d.text((10, H - 18), "side view   (green = foot on ground)", fill=(0, 0, 0))
    writer.append_data(np.asarray(frame))
writer.close()
print(f"saved {out}: {ep} episode(s), {falls} fall(s); current episode forward {last[1] - start[1]:+.2f} m", flush=True)
