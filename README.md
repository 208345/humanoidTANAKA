# humanoid

3人チームで開発する小型ヒューマノイドロボット。二足歩行を目標とし、
シミュレーション上で強化学習した歩行ポリシーの実機転移（sim-to-real）を目指す。

---

## 設計の中心にある一つの約束

**制御コードは、相手がシミュレータなのか実機なのかを知らない。**

```
Controller ──→ RobotInterface ──┬──→ SimRobot    (MuJoCo)
(手設計 or 学習)                 ├──→ RealRobot   (シリアル/UDP)
                                └──→ DummyRobot  (テスト用)
```

切り替えは起動時の引数一つ。この約束を守る限り、
sim で歩けたコードはそのまま実機で動かせる。**この境界を壊す変更は原則マージしない。**

同様に、手で設計した制御器と学習したポリシーも `Controller` として差し替え可能にしてある。
比較実験がそのまま実行できる。

---

## クイックスタート（Windows / PowerShell）

```powershell
git clone <このリポジトリのURL>
cd humanoid

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` が必要な場合があります。

動作確認：

```powershell
python -m control.run --backend dummy --controller zero --steps 100
pytest
```

`dummy` バックエンドは物理シミュレーション無しで動くので、
MuJoCo の環境構築前でもインターフェースの動作を確認できます。

シミュレータを使う場合：

```powershell
pip install -e ".[sim]"
python -m control.run --backend sim --controller zero --steps 1000
```

---

## ディレクトリ

| パス | 中身 | 主担当 |
|---|---|---|
| `model/` | **寸法・質量の唯一の正**。URDF / MJCF / params.yaml | 全員 |
| `docs/interfaces/` | **関節定義・座標系・通信仕様**。コードより先に決める | 全員 |
| `docs/adr/` | 設計判断の記録（なぜそうしたか） | 全員 |
| `firmware/` | マイコン側。サーボ制御ループ（1kHz） | A |
| `control/interface/` | sim / 実機 / ポリシーの差し替え層 | 全員 |
| `control/kinematics/` | 順運動学・逆運動学 | B |
| `control/estimator/` | 状態推定（IMU＋関節角 → 重心・ZMP） | C |
| `control/stabilizer/` | バランス制御（200〜500Hz） | B |
| `control/gait/` | 歩行パターン生成（50Hz） | B |
| `control/planner/` | 歩容計画（10Hz） | B |
| `learning/` | 強化学習の環境定義・学習・評価 | C |
| `sim/` | シミュレーション設定とシナリオ | C |
| `tools/` | ログ可視化、キャリブレーション | C |
| `hardware/` | 図面、部品表、配線図 | A |
| `datasets/` | 実機ログ（大きいものは Git に入れない） | - |

`control/` の分割が細かいのは、歩行制御が**周期の違う階層の積み重ね**だから。
ディレクトリがそのまま設計図になっている。

---

## 進行順（各段階が完了したらタグを打つ）

- [ ] **Phase 0** — `model/params.yaml` と `docs/interfaces/` を3人で合意して確定
- [ ] **Phase 1** — sim で全関節が指令通り動く
- [ ] **Phase 2** — 実機で全関節が指令通り動く（**Phase 1 と同じ Controller コードで**）
- [ ] **Phase 3** — sim で片足立ち（バランス制御）
- [ ] **Phase 4** — sim で歩行（まず既存の小型ヒューマノイドモデルで、次に自機で）
- [ ] **Phase 5** — 実機で歩行（sim-to-real）

Phase 1 と 2 の間で「同じ Controller が動く」ことを守れるかが最初の関門。
ここを雑にすると Phase 4 から 5 に進めなくなる。

Phase 5 は難関であり、到達できない可能性がある。
**その場合でも「なぜ転移しなかったか」を計測して記録することがこのプロジェクトの成果になる。**

---

## チームのルール

詳細は [CONTRIBUTING.md](CONTRIBUTING.md)。要点だけ：

- `main` へ直接 push しない。必ずブランチ → Pull Request → 他の1人がレビュー
- `main` は常に動く状態を保つ
- `model/` と `docs/interfaces/` の変更は**全員のレビュー必須**
- 大きいファイル（CAD、動画、ログ）は Git に入れない
