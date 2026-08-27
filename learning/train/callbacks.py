"""SB3 用のカスタムコールバック。

チェックポイント保存、VecNormalize 統計保存、カリキュラム進行を
学習ループに統合する。
"""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from learning.train.curriculum import CurriculumManager
from learning.train.rewards import StandingReward, WalkingReward


class CheckpointCallback(BaseCallback):
    """定期的にチェックポイントを保存する。

    VecNormalize の統計も一緒に保存し、Resume 時に復元できるようにする。
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "learning/policies/latest",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_k = self.num_timesteps // 1000
            model_path = self.save_path / f"{step_k}k.zip"
            self.model.save(str(model_path))

            # VecNormalize の統計も保存
            if isinstance(self.training_env, VecNormalize):
                norm_path = self.save_path / f"{step_k}k_vecnorm.pkl"
                self.training_env.save(str(norm_path))

            if self.verbose > 0:
                print(f"\n[SAVE] チェックポイント保存: {step_k}k.zip")

        return True


class CurriculumCallback(BaseCallback):
    """カリキュラム学習の進行を管理するコールバック。

    エピソード終了時の報酬を記録し、卒業条件を満たしたら
    次のフェーズに自動遷移する。
    """

    def __init__(
        self,
        curriculum: CurriculumManager,
        env,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.curriculum = curriculum
        self._env = env  # 報酬関数の差し替えに使用

    def _on_step(self) -> bool:
        self.curriculum.update_steps(1)

        # エピソード終了時の報酬を記録
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.curriculum.record_episode(ep_reward)

        # 卒業判定
        if self.curriculum.should_graduate():
            new_phase = self.curriculum.advance()
            self._apply_phase(new_phase)

        return True

    def _apply_phase(self, phase) -> None:
        """新しいフェーズの設定を環境に適用する。"""
        # 報酬関数の差し替え
        reward_config = self.curriculum.get_reward_params()
        reward_type = reward_config["type"]
        reward_params = reward_config["params"]

        if reward_type == "standing":
            reward_fn = StandingReward(**reward_params)
        elif reward_type == "walking":
            reward_fn = WalkingReward(**reward_params)
        else:
            raise ValueError(f"未知の報酬タイプ: {reward_type}")

        self._env.set_reward_fn(reward_fn)

        if self.verbose > 0:
            print(f"  報酬関数: {reward_type}")
            print(f"  環境パラメータ: {phase.env_overrides}")