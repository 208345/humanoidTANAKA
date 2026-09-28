"""設計を変えずに「両足を交互に浮かせる」周期動作が存在するかを探索する。

    python -m learning.eval.gait_search [n_random] [n_refine] [out_dir]

見つけた最良パラメータを <out_dir>/gait_best.npy に保存する（model/reference_gait.yaml に転記して使う）。
2026-09-26 の実行（20000 / 300）で得た解が model/reference_gait.yaml。

各脚の関節目標角（rad）:
  hip_roll_i  = s_i * r_add + r_amp * sin(w t + phr)        # 揺動は両脚同指令（軸が非ミラー）
  hip_pitch_i = m_i * (p_off + p_amp * sin(w t + th_i))
  knee_i      = m_i * (k_off + k_amp * max(0, sin(w t + th_i + phk)))
  ankle_i     = m_i * (a_off + a_amp * sin(w t + th_i + pha))
  L: s=+1, m=+1, th=0 / R: s=-1, m=-1, th=pi
"""
import sys, math
import numpy as np
from multiprocessing import Pool

LO = np.array([0.3, -0.3, 0.0, -np.pi, -0.6, -0.6, -0.4, 0.0, -np.pi, -0.6, -0.6, -np.pi])
HI = np.array([1.0,  0.3, 0.5,  np.pi,  0.6,  0.6,  1.0, 1.2,  np.pi,  0.6,  0.6,  np.pi])
NAMES = ["T", "r_add", "r_amp", "phr", "p_off", "p_amp", "k_off", "k_amp", "phk", "a_off", "a_amp", "pha"]

_env = None


def targets(x, t):
    T, r_add, r_amp, phr, p_off, p_amp, k_off, k_amp, phk, a_off, a_amp, pha = x
    w = 2 * math.pi / T
    out = np.zeros(8)
    for leg, (s, m, th) in enumerate([(1, 1, 0.0), (-1, -1, math.pi)]):
        o = leg * 4
        out[o + 0] = s * r_add + r_amp * math.sin(w * t + phr)
        out[o + 1] = m * (p_off + p_amp * math.sin(w * t + th))
        out[o + 2] = m * (k_off + k_amp * max(0.0, math.sin(w * t + th + phk)))
        out[o + 3] = m * (a_off + a_amp * math.sin(w * t + th + pha))
    return np.clip(out, -1.5, 1.5)


def rollout(x, seconds=8.0, record=False, seed=0):
    global _env
    if _env is None:
        from learning.envs.pybullet_env import PyBulletHumanoidEnv
        _env = PyBulletHumanoidEnv("model/humanoid.urdf")
        _env.action_scales[:] = 1.0            # 目標角を rad でそのまま与える
    env = _env
    p = env._p
    env.reset(seed=seed)
    rid = env._robot_id
    y0 = p.getBasePositionAndOrientation(rid)[0][1]
    n = int(seconds * env.control_hz)
    lift = np.zeros(2)
    counted = 0
    alive = 0
    log = []
    for k in range(n):
        t = k * env.control_dt
        # 最初の 0.5s で姿勢オフセットへ滑らかに移行
        ramp = min(1.0, t / 0.5)
        env.step(targets(x, t) * ramp)
        st = env._read_state()
        _, o = p.getBasePositionAndOrientation(rid)
        e = p.getEulerFromQuaternion(o)
        if st["body_contact"] > 0 or st["base_height"] < 0.10 or abs(e[0]) > 0.8 or abs(e[1]) > 0.8:
            break
        alive = k + 1
        if t > 1.0:
            counted += 1
            lift += ((1 - st["foot_contacts"]) * (st["foot_heights"] - 0.014 > 0.005)).astype(float)
        if record:
            log.append((t, *st["foot_contacts"], *(st["foot_heights"] - 0.014), st["base_height"]))
    y1 = p.getBasePositionAndOrientation(rid)[0][1]
    frac = lift / max(counted, 1)
    surv = alive / n
    # 生存していないと両足リフトは無効。min(L,R) が本命
    v = (y1 - y0) / (alive * env.control_dt) if alive else 0.0
    score = surv * (min(frac) + 0.05 * float(np.mean(frac)) + 0.3 * float(np.clip(v, 0.0, 0.2)) / 0.2) + 0.3 * surv
    res = dict(score=score, liftL=frac[0], liftR=frac[1], surv=surv, fwd=y1 - y0)
    if record:
        res["log"] = log
    return res


def evaluate(x):
    # 初期ノイズ違いの 3 回で最悪値を採る（ナイフエッジ解を避ける）
    rs = [rollout(np.asarray(x), seed=s) for s in (0, 1, 2)]
    return min(rs, key=lambda r: r["score"])


def main():
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_ref = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = np.random.default_rng(0)
    X = LO + (HI - LO) * rng.random((n_rand, len(LO)))
    with Pool(12) as pool:
        R = pool.map(evaluate, X, chunksize=20)
        order = np.argsort([-r["score"] for r in R])
        print("=== random search top 5 ===", flush=True)
        for i in order[:5]:
            r = R[i]
            print(f"score {r['score']:.3f} liftL {r['liftL']:.0%} liftR {r['liftR']:.0%} surv {r['surv']:.0%} fwd {r['fwd']:+.2f}", flush=True)
        survived = sum(r["surv"] > 0.99 for r in R)   # 8s 完走数
        both = sum(r["surv"] > 0.99 and min(r["liftL"], r["liftR"]) > 0.05 for r in R)
        print(f"survived 8s: {survived}/{n_rand}, both feet lift >5% of time: {both}/{n_rand}", flush=True)

        # 上位から局所改良（各世代 12 個の摂動）
        best_x, best = X[order[0]].copy(), R[order[0]]
        sigma = 0.15
        for g in range(n_ref):
            cand = np.clip(best_x + sigma * (HI - LO) * rng.standard_normal((12, len(LO))), LO, HI)
            cr = pool.map(evaluate, cand)
            j = int(np.argmax([c["score"] for c in cr]))
            if cr[j]["score"] > best["score"]:
                best_x, best = cand[j], cr[j]
            else:
                sigma = max(sigma * 0.93, 0.02)   # 幅を潰しきらない
            if g % 10 == 0 or g == n_ref - 1:
                print(f"refine {g:3d}: score {best['score']:.3f} liftL {best['liftL']:.0%} liftR {best['liftR']:.0%} "
                      f"surv {best['surv']:.0%} fwd {best['fwd']:+.2f} sigma {sigma:.3f}", flush=True)
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    np.save(f"{out_dir}/gait_best.npy", best_x)
    np.save(f"{out_dir}/gait_top_random.npy", X[order[:20]])
    print("best params:", {n: round(float(v), 3) for n, v in zip(NAMES, best_x)}, flush=True)


if __name__ == "__main__":
    main()
