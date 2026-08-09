# 開発ルール

3人で並行作業するための取り決め。**技術より先にこれを守ることが、プロジェクトが完走する条件。**

---

## 1. ブランチとレビュー

`main` への直接 push は禁止（GitHub の Branch protection で設定する。手順は下記）。

```powershell
git switch -c feat/ik-solver     # ブランチを切る
# ... 作業 ...
git add -A
git commit -m "逆運動学ソルバの実装"
git push -u origin feat/ik-solver
```

GitHub 上で Pull Request を作成 → **他の1人がレビュー** → マージ。

ブランチ名の接頭辞：

| 接頭辞 | 用途 |
|---|---|
| `feat/` | 機能追加 |
| `fix/` | バグ修正 |
| `docs/` | ドキュメントのみ |
| `exp/` | 実験・お試し（マージしないこともある） |

### Branch protection の設定（最初に1回、リポジトリ作成者が行う）

GitHub のリポジトリページ → Settings → Rules → Rulesets → New branch ruleset

- Target branches に `main` を指定
- **Require a pull request before merging** を有効化（Required approvals: 1）
- **Require status checks to pass** を有効化し、CI を選択

---

## 2. レビューで必ず見ること

- `control/interface/` の境界を壊していないか（**最優先**）
- 単位は SI か（角度は **rad**、長さは **m**、時間は **秒**）
- `model/params.yaml` の値をコード内にベタ書きしていないか
- 新しい定数に根拠のコメントがあるか

レビューは「粗探し」ではなく**他人のコードを読む練習**。
分からない箇所は遠慮なく質問を書く。それがレビューの主目的。

---

## 3. 変更に全員の合意が要るもの

以下は1人で決めない。Issue で議論してから変更する。

- `model/params.yaml`（寸法・質量・可動域）
- `docs/interfaces/` 配下すべて（関節定義・座標系・通信仕様）
- `control/interface/robot_interface.py` の抽象クラス定義

**ここが3人の作業をつなぐ唯一の接点**なので、勝手に変えると他の2人の作業が壊れる。

---

## 4. 設計判断を記録する（ADR）

「なぜこうしたか」は3ヶ月後に必ず忘れる。重要な判断をしたら `docs/adr/` に1ファイル残す。

`docs/adr/0000-template.md` をコピーして番号を振る。5分で書ける分量でよい。

---

## 5. Issue の使い方

作業を始める前に Issue を立てて自分にアサインする。
「今誰が何をしているか」が見えないことが、3人開発で最も事故が起きる原因。

---

## 6. Git に入れないもの

`.gitignore` で除外済み。うっかり追加しないこと。

- `.venv/`、`__pycache__/`
- CAD の生ファイル、動画、実機ログ（Google Drive など別の場所へ）
- 学習済みモデルの重み（大きいものは GitHub Releases か外部ストレージへ）

一度コミットすると履歴に残り続け、後から消すのが非常に面倒。**push 前に `git status` を確認する習慣をつける。**
