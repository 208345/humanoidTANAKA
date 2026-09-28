# policies — 学習済みモデル

学習中のチェックポイントは大量・大容量なので Git に入れない（.gitignore で除外）。
代表的な小さいモデル（数MB）だけ `git add -f` でこのフォルダに入れ、ここに索引を書く。

| 名前 | 学習日 | 環境のコミット | sim成績 | 実機成績 | 場所 |
|---|---|---|---|---|---|
| res1_best（参照歩容 + 残差 PPO, 22.8M step） | 2026-09-26 | 621d033 | PyBullet 20s×10: 転倒 0/10、0.128 m/s、両足交互離地 (L34%/R27%)、3N 外乱でも転倒 0/10 | 未 | `learning/policies/res1_best/best.zip` + `best_vecnorm.pkl`（リポジトリ内） |

### res1_best の再現・確認方法

`model/params_residual.yaml` と組で使う（行動 = `model/reference_gait.yaml` の参照歩容からの残差）。

```bash
# 学習（参照歩容の安定化）
python -m learning.train.train --urdf model/humanoid.urdf \
  --params model/params_residual.yaml --curriculum learning/train/curriculum_residual.yaml \
  --reward-type walking --log-std-init -1.0 --num-envs 12 --total-timesteps 25000000 --save-path <out>

# 数値評価 / 動画
python -m learning.eval.gait_metrics --urdf model/humanoid.urdf --model <ckpt>.zip --vecnorm <ckpt>_vecnorm.pkl \
  --params model/params_residual.yaml --curriculum learning/train/curriculum_residual.yaml --episodes 10
python -m learning.eval.record_video <ckpt(拡張子なし)> out.mp4 40 --residual
# 例: python -m learning.eval.record_video learning/policies/res1_best/best res1.mp4 40 --residual
```

参照歩容だけ（学習なし）でも歩くが、フィードバックが無いので 20s で 9/10 転倒・大きく曲がる。
参照歩容は `learning/eval/gait_search.py` で探索したもの（式は `model/reference_gait.yaml` の先頭コメント）。

**環境のコミットハッシュを必ず残すこと。** 報酬設計を変えた後では、
どの環境で学習した重みなのかが分からなくなる。
