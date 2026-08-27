"""カリキュラム学習の管理。

cube-sim-rl では手動で行っていたカリキュラム遷移を自動化する。
YAML 設定ファイルからフェーズ定義を読み込み、学習中の卒業判定を行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CurriculumPhase:
    """カリキュラムの1段階の定義。"""

    name: str
    reward_type: str = "standing"  # "standing" or "walking"
    reward_params: dict = field(default_factory=dict)

    # 環境パラメータの上書き
    env_overrides: dict = field(default_factory=dict)

    # 卒業条件
    graduation_metric: str = "mean_reward"
    graduation_threshold: float = 5.0
    graduation_window: int = 50  # 直近 N エピソードの平均
    min_steps: int = 50000       # 最低学習ステップ数


class CurriculumManager:
    """カリキュラムの進行管理。

    SB3 の Callback 内から呼ばれ、学習の進捗に応じて
    フェーズの遷移と環境パラメータの更新を行う。

    Args:
        config_path: カリキュラム定義 YAML のパス。
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.phases = self._load_config()
        self.current_phase_idx = 0
        self._episode_rewards: list[float] = []
        self._total_steps = 0

    def _load_config(self) -> list[CurriculumPhase]:
        with open(self.config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        phases = []
        for p in config.get("phases", []):
            phases.append(CurriculumPhase(
                name=p["name"],
                reward_type=p.get("reward_type", "standing"),
                reward_params=p.get("reward_params", {}),
                env_overrides=p.get("env_overrides", {}),
                graduation_metric=p.get("graduation", {}).get("metric", "mean_reward"),
                graduation_threshold=p.get("graduation", {}).get("threshold", 5.0),
                graduation_window=p.get("graduation", {}).get("window", 50),
                min_steps=p.get("graduation", {}).get("min_steps", 50000),
            ))
        return phases

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_phase_idx]

    @property
    def is_final_phase(self) -> bool:
        return self.current_phase_idx >= len(self.phases) - 1

    def record_episode(self, reward: float) -> None:
        """エピソード終了時の報酬を記録する。"""
        self._episode_rewards.append(reward)

    def update_steps(self, n_steps: int) -> None:
        """学習ステップ数を更新する。"""
        self._total_steps += n_steps

    def should_graduate(self) -> bool:
        """現在のフェーズを卒業すべきか判定する。"""
        if self.is_final_phase:
            return False

        phase = self.current_phase

        # 最低ステップ数の確認
        if self._total_steps < phase.min_steps:
            return False

        # 直近 N エピソードの平均報酬
        window = phase.graduation_window
        if len(self._episode_rewards) < window:
            return False

        recent = self._episode_rewards[-window:]
        mean_reward = sum(recent) / len(recent)

        return mean_reward >= phase.graduation_threshold

    def advance(self) -> CurriculumPhase:
        """次のフェーズに進む。"""
        if self.is_final_phase:
            raise RuntimeError("最終フェーズです。これ以上進めません。")

        self.current_phase_idx += 1
        self._episode_rewards.clear()
        self._total_steps = 0

        phase = self.current_phase
        print(f"\n{'='*60}")
        print(f"🎓 カリキュラム昇格: {phase.name}")
        print(f"{'='*60}\n")

        return phase

    def get_reward_params(self) -> dict[str, Any]:
        """現在のフェーズの報酬関数パラメータを返す。"""
        return {
            "type": self.current_phase.reward_type,
            "params": self.current_phase.reward_params,
        }