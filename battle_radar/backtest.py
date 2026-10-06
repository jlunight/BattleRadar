from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from battle_radar.config import AppConfig, DEFAULT_CONFIG
from battle_radar.database import SQLiteStore
from battle_radar.utils import normalize_trade_date


logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, config: AppConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store or SQLiteStore(self.config.db_path)

    def _posterior_win_rate(self, success_count: int, sample_count: int) -> float:
        quality = self.config.strategy_quality
        numerator = success_count + quality.prior_success
        denominator = sample_count + quality.prior_success + quality.prior_failure
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _sample_level(self, sample_count: int, cycle_samples: int) -> str:
        quality = self.config.strategy_quality
        if sample_count < quality.min_band_samples:
            return "冷启动"
        if sample_count < quality.min_cycle_samples or cycle_samples < quality.min_band_samples:
            return "样本偏少"
        if sample_count < quality.min_strategy_samples or cycle_samples < quality.min_cycle_samples:
            return "观察期"
        return "样本充足"

    def _recommended_status(
        self,
        sample_count: int,
        bayesian_win_rate: float,
        recent_10_win_rate: float | None,
        recent_30_win_rate: float | None,
        best_cycle_samples: int,
    ) -> tuple[str, str]:
        quality = self.config.strategy_quality
        target = quality.target_win_rate_pct
        recent_floor = target - 5

        if sample_count < quality.min_band_samples:
            return "观察", "总样本过少，仍处冷启动"
        if sample_count < quality.min_strategy_samples or best_cycle_samples < quality.min_cycle_samples:
            return "观察", "历史样本未达启用门槛"
        if bayesian_win_rate >= target:
            recent_checks: list[float] = [value for value in [recent_10_win_rate, recent_30_win_rate] if value is not None]
            if not recent_checks or min(recent_checks) >= recent_floor:
                return "启用", "历史与近期表现同时达标"
            return "观察", "长期达标但近期表现回落"
        if bayesian_win_rate >= target - 10:
            return "观察", "接近目标胜率，等待更多样本确认"
        return "停用", "历史胜率显著低于目标"

    @staticmethod
    def _profit_loss_ratio(frame: pd.DataFrame) -> float | None:
        profits = frame.loc[frame["pnl_ratio"] > 0, "pnl_ratio"]
        losses = frame.loc[frame["pnl_ratio"] <= 0, "pnl_ratio"].abs()
        if profits.empty or losses.empty:
            return None
        avg_loss = losses.mean()
        if not avg_loss:
            return None
        return float(profits.mean() / avg_loss)

    def _diagnostic_frame(self) -> pd.DataFrame:
        sql = """
        SELECT
            b.signal_date,
            b.strategy_name,
            b.code,
            b.name,
            b.buy_mode,
            b.buy_price,
            b.next_trade_date,
            b.next_open,
            b.next_high,
            b.max_return_pct,
            b.success,
            b.pnl_ratio,
            b.sample_tag,
            COALESCE(s.cycle_label, '未知') AS cycle_label
        FROM backtest_results b
        LEFT JOIN strategy_signals s
          ON s.trade_date = b.signal_date
         AND s.strategy_name = b.strategy_name
         AND s.code = b.code
        ORDER BY b.signal_date, b.strategy_name, b.code
        """
        return self.store.query(sql)

    def _trade_calendar(self) -> list[str]:
        df = self.store.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date")
        return df["trade_date"].tolist() if not df.empty else []

    def _next_trade_date(self, trade_date: str) -> str | None:
        trade_date = normalize_trade_date(trade_date) or trade_date
        calendar = self._trade_calendar()
        if trade_date not in calendar:
            later = [d for d in calendar if d > trade_date]
            return later[0] if later else None
        idx = calendar.index(trade_date)
        return calendar[idx + 1] if idx + 1 < len(calendar) else None

    def backtest_signals(self, signal_date: str | None = None) -> pd.DataFrame:
        signal_date = normalize_trade_date(signal_date) if signal_date else None
        if signal_date:
            signals = self.store.query("SELECT * FROM strategy_signals WHERE trade_date = ?", (signal_date,))
        else:
            signals = self.store.query(
                """
                SELECT s.*
                FROM strategy_signals s
                LEFT JOIN backtest_results b
                  ON b.signal_date = s.trade_date
                 AND b.strategy_name = s.strategy_name
                 AND b.code = s.code
                WHERE b.signal_date IS NULL
                ORDER BY s.trade_date, s.strategy_name, s.code
                """
            )
        if signals.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for _, signal in signals.iterrows():
            next_trade_date = self._next_trade_date(signal["trade_date"])
            if not next_trade_date:
                continue
            next_price = self.store.query(
                """
                SELECT trade_date, code, open, high, close
                FROM daily_prices
                WHERE trade_date = ? AND code = ?
                """,
                (next_trade_date, signal["code"]),
            )
            if next_price.empty:
                continue
            next_row = next_price.iloc[0]
            if signal["buy_mode"] == "limit_up" and pd.notna(next_row["high"]):
                buy_price = float(next_row["high"])
            else:
                buy_price = float(next_row["open"])
            next_high = float(next_row["high"])
            next_close = float(next_row["close"])
            max_return_pct = round((next_high - buy_price) / buy_price * 100, 2) if buy_price else None
            close_return_pct = round((next_close - buy_price) / buy_price * 100, 2) if buy_price else None
            success = int((max_return_pct or -999) >= self.config.buy_rule.success_threshold_pct)
            pnl_ratio = self.config.buy_rule.success_threshold_pct if success else close_return_pct
            rows.append(
                {
                    "signal_date": signal["trade_date"],
                    "strategy_name": signal["strategy_name"],
                    "code": signal["code"],
                    "name": signal["name"],
                    "buy_mode": signal["buy_mode"],
                    "buy_price": buy_price,
                    "next_trade_date": next_trade_date,
                    "next_open": float(next_row["open"]),
                    "next_high": next_high,
                    "max_return_pct": max_return_pct,
                    "success": success,
                    "pnl_ratio": pnl_ratio,
                    "sample_tag": ">=4%达标" if success else "未达标",
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            self.store.upsert_dataframe(
                "backtest_results",
                result,
                unique_columns=["signal_date", "strategy_name", "code"],
            )
        logger.info("回测完成，新增结果: %s", len(result))
        return result

    def summarize_performance(self) -> pd.DataFrame:
        frame = self._diagnostic_frame()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "strategy_name",
                    "recommended_status",
                    "recommendation_reason",
                    "sample_level",
                    "sample_count",
                    "success_count",
                    "cumulative_win_rate",
                    "bayesian_win_rate",
                    "recent_10_win_rate",
                    "recent_30_win_rate",
                    "avg_max_return",
                    "profit_loss_ratio",
                    "best_cycle_label",
                    "best_cycle_samples",
                    "best_cycle_win_rate",
                    "best_cycle_bayesian_win_rate",
                ]
            )

        frame = frame.sort_values(["strategy_name", "signal_date", "code"]).reset_index(drop=True)
        rows: list[dict[str, Any]] = []

        for strategy_name, strategy_frame in frame.groupby("strategy_name", sort=False):
            strategy_frame = strategy_frame.sort_values(["signal_date", "code"]).reset_index(drop=True)
            sample_count = int(len(strategy_frame))
            success_count = int(strategy_frame["success"].fillna(0).sum())
            cumulative_win_rate = round(success_count / sample_count * 100, 2) if sample_count else None
            bayesian_win_rate = round(self._posterior_win_rate(success_count, sample_count) * 100, 2) if sample_count else None

            recent_10 = strategy_frame.tail(10)
            recent_30 = strategy_frame.tail(30)
            recent_10_win_rate = round(recent_10["success"].mean() * 100, 2) if not recent_10.empty else None
            recent_30_win_rate = round(recent_30["success"].mean() * 100, 2) if not recent_30.empty else None
            avg_max_return = round(float(strategy_frame["max_return_pct"].dropna().mean()), 2) if strategy_frame["max_return_pct"].notna().any() else None
            profit_loss_ratio = self._profit_loss_ratio(strategy_frame)

            cycle_rows: list[dict[str, Any]] = []
            for cycle_label, cycle_frame in strategy_frame.groupby("cycle_label", dropna=False, sort=False):
                cycle_sample_count = int(len(cycle_frame))
                cycle_success_count = int(cycle_frame["success"].fillna(0).sum())
                cycle_win_rate = round(cycle_success_count / cycle_sample_count * 100, 2) if cycle_sample_count else 0.0
                cycle_bayesian_win_rate = round(
                    self._posterior_win_rate(cycle_success_count, cycle_sample_count) * 100,
                    2,
                )
                cycle_rows.append(
                    {
                        "cycle_label": cycle_label or "未知",
                        "cycle_sample_count": cycle_sample_count,
                        "cycle_win_rate": cycle_win_rate,
                        "cycle_bayesian_win_rate": cycle_bayesian_win_rate,
                    }
                )

            cycle_summary = pd.DataFrame(cycle_rows).sort_values(
                ["cycle_bayesian_win_rate", "cycle_sample_count", "cycle_win_rate"],
                ascending=[False, False, False],
            )
            best_cycle = cycle_summary.iloc[0] if not cycle_summary.empty else {}
            best_cycle_label = best_cycle.get("cycle_label", "未知")
            best_cycle_samples = int(best_cycle.get("cycle_sample_count", 0) or 0)
            best_cycle_win_rate = best_cycle.get("cycle_win_rate")
            best_cycle_bayesian_win_rate = best_cycle.get("cycle_bayesian_win_rate")

            sample_level = self._sample_level(sample_count, best_cycle_samples)
            recommended_status, recommendation_reason = self._recommended_status(
                sample_count=sample_count,
                bayesian_win_rate=bayesian_win_rate or 0.0,
                recent_10_win_rate=recent_10_win_rate,
                recent_30_win_rate=recent_30_win_rate,
                best_cycle_samples=best_cycle_samples,
            )

            rows.append(
                {
                    "strategy_name": strategy_name,
                    "recommended_status": recommended_status,
                    "recommendation_reason": recommendation_reason,
                    "sample_level": sample_level,
                    "sample_count": sample_count,
                    "success_count": success_count,
                    "cumulative_win_rate": cumulative_win_rate,
                    "bayesian_win_rate": bayesian_win_rate,
                    "recent_10_win_rate": recent_10_win_rate,
                    "recent_30_win_rate": recent_30_win_rate,
                    "avg_max_return": avg_max_return,
                    "profit_loss_ratio": round(profit_loss_ratio, 2) if profit_loss_ratio is not None else None,
                    "best_cycle_label": best_cycle_label,
                    "best_cycle_samples": best_cycle_samples,
                    "best_cycle_win_rate": best_cycle_win_rate,
                    "best_cycle_bayesian_win_rate": best_cycle_bayesian_win_rate,
                }
            )

        summary = pd.DataFrame(rows)
        status_rank = {"启用": 0, "观察": 1, "停用": 2}
        summary["_status_rank"] = summary["recommended_status"].map(status_rank).fillna(9)
        summary = summary.sort_values(
            ["_status_rank", "bayesian_win_rate", "sample_count", "recent_10_win_rate"],
            ascending=[True, False, False, False],
        ).drop(columns="_status_rank")
        return summary.reset_index(drop=True)

    def latest_validation(self, trade_date: str) -> pd.DataFrame:
        trade_date = normalize_trade_date(trade_date) or trade_date
        sql = """
        SELECT *
        FROM backtest_results
        WHERE signal_date = ?
        ORDER BY success DESC, max_return_pct DESC
        """
        return self.store.query(sql, (trade_date,))


if __name__ == "__main__":
    engine = BacktestEngine()
    df = engine.backtest_signals()
    print(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2))
