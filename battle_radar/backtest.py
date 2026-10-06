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
        sql = """
        WITH base AS (
            SELECT
                strategy_name,
                signal_date,
                COUNT(*) AS samples,
                AVG(success) AS win_rate,
                AVG(CASE WHEN pnl_ratio > 0 THEN pnl_ratio END) AS avg_profit,
                AVG(CASE WHEN pnl_ratio <= 0 THEN ABS(pnl_ratio) END) AS avg_loss
            FROM backtest_results
            GROUP BY strategy_name, signal_date
        ),
        last_30 AS (
            SELECT strategy_name,
                   AVG(success) AS rolling_30d_win_rate
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY strategy_name ORDER BY signal_date DESC) AS rn
                FROM backtest_results
            )
            WHERE rn <= 30
            GROUP BY strategy_name
        )
        SELECT
            b.strategy_name,
            SUM(b.samples) AS sample_count,
            ROUND(AVG(b.win_rate) * 100, 2) AS cumulative_win_rate,
            ROUND(AVG(CASE WHEN b.avg_loss > 0 THEN b.avg_profit / b.avg_loss END), 2) AS profit_loss_ratio,
            ROUND(MAX(l.rolling_30d_win_rate) * 100, 2) AS rolling_30d_win_rate
        FROM base b
        LEFT JOIN last_30 l
          ON l.strategy_name = b.strategy_name
        GROUP BY b.strategy_name
        ORDER BY cumulative_win_rate DESC, sample_count DESC
        """
        return self.store.query(sql)

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
