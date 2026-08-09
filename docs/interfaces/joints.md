# 関節定義

> **Status: 未確定（Phase 0 で3人で確定させる）**

## 共通ルール

- 単位は **rad**（degree は使わない。表示するときだけ変換する）
- 配列の並び順は下の表の ID 順。`model/params.yaml` の `joints` と**必ず一致させる**
- 符号は**右ねじ**（軸の正方向に対して反時計回りが正）

## 原点姿勢（zero pose）

全関節が 0 [rad] のときの姿勢を、文章と写真の両方で定義する。

> TODO: 直立し、脚をまっすぐ伸ばし、つま先を正面に向けた姿勢を原点とする（要合意）
> TODO: 実機を組んだら原点姿勢の写真を `docs/images/zero_pose.jpg` に置く

**キャリブレーションはこの姿勢を基準に行う。** 実機の原点がずれていると、
sim で学習したポリシーは絶対に転移しない。

## 関節一覧（下半身12自由度の案）

| ID | 名前 | 部位 | 回転軸 | 正方向 | 可動域 [deg] |
|---|---|---|---|---|---|
| 0 | `l_hip_yaw` | 左股 | Z | TODO | TODO |
| 1 | `l_hip_roll` | 左股 | X | TODO | TODO |
| 2 | `l_hip_pitch` | 左股 | Y | TODO | TODO |
| 3 | `l_knee_pitch` | 左膝 | Y | TODO | TODO |
| 4 | `l_ankle_pitch` | 左足首 | Y | TODO | TODO |
| 5 | `l_ankle_roll` | 左足首 | X | TODO | TODO |
| 6 | `r_hip_yaw` | 右股 | Z | TODO | TODO |
| 7 | `r_hip_roll` | 右股 | X | TODO | TODO |
| 8 | `r_hip_pitch` | 右股 | Y | TODO | TODO |
| 9 | `r_knee_pitch` | 右膝 | Y | TODO | TODO |
| 10 | `r_ankle_pitch` | 右足首 | Y | TODO | TODO |
| 11 | `r_ankle_roll` | 右足首 | X | TODO | TODO |

> 上半身は Phase 5 以降。まず12自由度で歩かせることに集中する。
> 自由度を増やすほど学習も実装も難しくなる。

## サーボの割り当て

| 関節ID | サーボID | 型番 | 減速比 | 備考 |
|---|---|---|---|---|
| 0 | TODO | TODO | TODO | |

> **選定時の必須条件**（機械学習をやる上での前提）
> - 現在角度が読めること
> - 現在電流（またはトルク）が読めること
> - 50Hz 以上で読み書きできること
>
> ログが取れない機体では、学習も原因分析もできない。ここは妥協しないこと。
