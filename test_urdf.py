import pybullet as p
import pybullet_data
import yaml
import time
import sys

def main():
    print("--- URDF 読み込みテスト ---")
    
    # 1. params.yaml の確認
    try:
        with open("model/params.yaml", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        num_joints_expected = params["robot"]["num_joints"]
        joint_names_expected = [j["name"] for j in params["joints"]]
        print(f"params.yaml: 関節数 {num_joints_expected}")
    except Exception as e:
        print(f"params.yaml の読み込みエラー: {e}")
        return

    # 2. PyBullet の起動
    p.connect(p.DIRECT)  # DIRECT でGUIを出さずにテスト
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 3. URDF の読み込み
    try:
        robot_id = p.loadURDF("model/humanoid.urdf", basePosition=[0, 0, 0.5], useFixedBase=False)
        print("URDF の読み込みに成功しました！")
    except Exception as e:
        print(f"URDF の読み込みエラー: {e}")
        return
        
    # 4. 関節の確認
    num_joints_actual = p.getNumJoints(robot_id)
    controllable_joints = []
    actual_joint_names = []
    
    for i in range(num_joints_actual):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        
        # JOINT_REVOLUTE(0) or JOINT_PRISMATIC(1)
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            controllable_joints.append(i)
            actual_joint_names.append(joint_name)
            
    print(f"\nURDF 内の制御可能な関節数: {len(controllable_joints)}")
    print("URDF 内の関節名:")
    for name in actual_joint_names:
        print(f"  - {name}")
        
    if len(controllable_joints) != num_joints_expected:
        print(f"\n[エラー] 関節数が params.yaml ({num_joints_expected}個) と一致しません！")
        sys.exit(1)
    else:
        print("\n[OK] 関節数は一致しています。")
        
    p.disconnect()

if __name__ == "__main__":
    main()
