from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class EmotionThresholds:
    ice_point_score: int = 20
    repair_score: int = 40
    rising_score: int = 60
    climax_score: int = 80
    board_count_high_quantile: float = 0.8
    board_height_high_quantile: float = 0.8
    amount_high_quantile: float = 0.8


@dataclass(slots=True)
class BuyRuleConfig:
    success_threshold_pct: float = 4.0
    default_buy_mode: str = "open"
    position_per_stock_limit: float = 0.30
    total_position_limit: float = 0.80
    stop_profit_pct: float = 4.0
    stop_loss_pct: float = -100.0


@dataclass(slots=True)
class StrategyQualityConfig:
    target_win_rate_pct: float = 80.0
    allow_cold_start: bool = False
    min_band_samples: int = 3
    min_cycle_samples: int = 5
    min_strategy_samples: int = 8
    prior_success: float = 1.0
    prior_failure: float = 1.0
    min_score_for_fallback: float = 90.0
    max_signals_per_strategy: int = 1
    max_signals_per_day: int = 2


@dataclass(slots=True)
class AppConfig:
    workspace_dir: Path = Path("/workspace")
    data_dir: Path = Path("/workspace/data")
    reports_dir: Path = Path("/workspace/reports")
    db_path: Path = Path("/workspace/data/battle_radar.db")
    log_level: str = "INFO"
    request_timeout: int = 20
    request_retries: int = 3
    backfill_start_date: str = "20240101"
    universe_batch_size: int = 200
    board_sample_size: int = 30
    emotion_thresholds: EmotionThresholds = field(default_factory=EmotionThresholds)
    buy_rule: BuyRuleConfig = field(default_factory=BuyRuleConfig)
    strategy_quality: StrategyQualityConfig = field(default_factory=StrategyQualityConfig)


DEFAULT_CONFIG = AppConfig()
