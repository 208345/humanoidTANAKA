with open('learning/train/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_calc = '''        # 2. 速度追従報酬（目標速度との差が小さいほど高報酬）
        # raw_state に base_velocity が含まれている場合に使用
        # 含まれていない場合は角速度から推定
        # TODO: PyBullet の getBaseVelocity() を raw_state に追加する
        vel_reward = 0.0

        reward += vel_reward * self.velocity_weight'''
new_calc = '''        # 2. 速度追従報酬（目標速度との差が小さいほど高報酬）
        base_vel = raw_state.get("base_linear_velocity", [0.0, 0.0, 0.0])
        # X方向（前方向）の速度を取得
        forward_vel = base_vel[0]
        # 目標速度との誤差をペナルティとして計算
        vel_error = abs(forward_vel - self.target_velocity)
        # 誤差が小さいほど高い報酬（最大1.0）になるように指数関数を使用
        import math
        vel_reward = math.exp(-2.0 * vel_error)

        reward += vel_reward * self.velocity_weight'''

content = content.replace(old_calc, new_calc)

with open('learning/train/rewards.py', 'w', encoding='utf-8') as f:
    f.write(content)
