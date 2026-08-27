# Unity シミュレーション セットアップガイド

このドキュメントは、**Unity担当者（シミュレーション環境構築担当）**向けの手順書です。
Python/MLチームが強化学習を行うためのUnity環境（実行ファイル）を構築する手順を説明します。

## 1. 必要な環境
- **Unity 2022.3 LTS** 以上（推奨）
- **ML-Agents Release 20** 以上 (Package Managerからインストール)
- **URDF Importer** (Package Managerからインストール)

## 2. プロジェクトの準備
1. Unity Hubから新しい **3D (URP または Standard)** プロジェクトを作成します。
2. Window > Package Manager を開き、以下のパッケージをインストールします：
   - ML Agents
   - URDF Importer (見つからない場合は + > Add package from git URL で com.unity.robotics.urdf-importer を入力)

## 3. ロボットモデルのインポート
1. Pythonチームから共有された humanoid.urdf と、必要な stl ファイルをUnityプロジェクトの Assets フォルダにドラッグ＆ドロップします。
2. humanoid.urdf を右クリックし、Import Robot from URDF を選択します。
3. インポート設定で、**Axis Type** を Y is Up に設定し、**VHACD**（凸包メッシュ生成）を有効にすることをおすすめします。
4. インポートされたプレハブ（Prefab）をシーンに配置します。

## 4. ArticulationBody の設定
URDF Importerを使用すると、各リンクに自動的に ArticulationBody がアタッチされます。
- ルート（Torso等）の ArticulationBody は **Immovable** のチェックを **外して** ください。
- 全ての関節（joint_1 〜 joint_6）の ArticulationBody において、以下の設定を確認してください：
  - **Articulation Joint Type**: Revolute または Prismatic (ヒューマノイドの関節は通常Revolute)
  - **Stiffness**, **Damping**, **Force Limit** を適切に設定します（Stiffness: 10000, Damping: 1000, Force Limit: 10000 など、ロボットの重量に合わせて調整）。

## 5. ML-Agents スクリプトのアタッチ
1. リポジトリ内の sim/unity/HumanoidAgent.cs をUnityプロジェクトの Assets/Scripts/ 等にインポートします。
2. シーン上のロボットのルートオブジェクト（一番上の親）に、以下の2つのコンポーネントをアタッチします：
   - HumanoidAgent (先ほどのスクリプト)
   - Decision Requester (ML-Agentsに同梱)
3. HumanoidAgent コンポーネントのインスペクター設定：
   - **Torso**: ルートオブジェクトの ArticulationBody をドラッグ＆ドロップ
   - **Joints**: joint_1 から joint_6 までの ArticulationBody を順番にドラッグ＆ドロップ
   - **Max Angle Deg**: 90 (デフォルト)
4. ロボットオブジェクトに紐付いている Behavior Parameters コンポーネントの設定：
   - **Behavior Name**: HumanoidAgent (重要: Python側と一致させる必要があります)
   - **Vector Observation > Space Size**: **19** (重要: 関節角度6+速度6+胴体姿勢4+胴体角速度3)
   - **Actions > Continuous Actions**: **6**
   - **Actions > Discrete Branches**: 0

## 6. Pythonチームへ渡すためのビルド (.exe)
Pythonチームがこの環境を使って強化学習を回せるように、スタンドアロンの実行ファイルを作成します。

1. File > Build Settings を開きます。
2. **Platform** を Windows, Mac, Linux に設定します。
3. Development Build のチェックは外しても構いません。
4. **Resolution and Presentation** (Player Settings) で、Run In Background を **有効** にしてください（非常に重要です）。
5. Build をクリックし、humanoid.exe などの名前で出力します。
6. 出力された実行ファイル一式（.exe本体と、同じフォルダにあるDataフォルダ等の全て）をZIP等で固めてPythonチームに共有してください！

以上でUnity側のセットアップは完了です！
