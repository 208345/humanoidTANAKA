# Unity担当者向け：強化学習（ML-Agents）セットアップガイド

Python側の強化学習AIと、Unityのシミュレーションを連携させるための手順書です。
**難しいプログラミングやビルド作業は一切不要です！** 以下の手順でシーンを設定するだけで、AIがUnity上のロボットを動かし始めます。

## 1. 準備するパッケージ
Unityプロジェクトを開き、Package Managerから以下をインストールしてください。
- **ML Agents**
- **URDF Importer**

## 2. ロボットの配置
1. 配布された humanoid.urdf を Unity にインポートしてシーンに置きます。
2. インポートされたロボットの全関節（Torsoやjoint_1〜6）に ArticulationBody が付いていることを確認してください。
   - ※ルート（Torso）の ArticulationBody は「Immovable」のチェックを**外して**ください（宙に浮かせるため）。

## 3. AIスクリプトのアタッチ
リポジトリ内の sim/unity/HumanoidAgent.cs をUnityにインポートし、**ロボットの親オブジェクト**にアタッチしてください。
インスペクター（設定画面）で以下の項目を埋めます：
- **Torso**: ロボットの胴体（ルート）のオブジェクトをドラッグ＆ドロップ
- **Joints**: joint_1 から joint_6 までの関節オブジェクトを順番にドラッグ＆ドロップ

※アタッチすると自動的に Behavior Parameters や Decision Requester という項目も追加されますが、そのままの設定で大丈夫です！

## 4. 学習のスタート！
セットアップはこれだけです！あとは以下の手順でAIと通信します。

1. **Unityの再生ボタン（▶）を押す**
2. Python環境が使えるターミナルを開き、以下のコマンドを実行する
   `powershell
   python -m learning.train.train --backend unity
   `

これだけで、Unity上のロボットがAIによってガシャガシャと動き出し、強化学習がスタートします！
