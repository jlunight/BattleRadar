from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta

import pandas as pd

from battle_radar.config import AppConfig, DEFAULT_CONFIG
from battle_radar.database import SQLiteStore


logger = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(self, config: AppConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store or SQLiteStore(self.config.db_path)

    def _trade_calendar(self) -> list[str]:
        df = self.store.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date")
        return df["trade_date"].tolist() if not df.empty else []

    def _previous_trade_date(self, trade_date: str) -> str | None:
        calendar = self._trade_calendar()
        if trade_date not in calendar:
            earlier = [d for d in calendar if d < trade_date]
            return earlier[-1] if earlier else None
        idx = calendar.index(trade_date)
        return calendar[idx - 1] if idx > 0 else None

    def _load_trade_frame(self, trade_date: str) -> pd.DataFrame:
        sql = """
        WITH board_strength AS (
            SELECT board_name, MAX(limit_up_count) AS board_limit_up_count, MAX(pct_chg) AS board_pct_chg
            FROM board_snapshot
            WHERE trade_date = ?
            GROUP BY board_name
        )
        SELECT
            lup.trade_date,
            lup.code,
            lup.name,
            lup.latest_price,
            lup.pct_chg,
            lup.amount,
            lup.circ_market_cap,
            lup.total_market_cap,
            lup.turnover,
            lup.seal_amount,
            lup.first_limit_time,
            lup.last_limit_time,
            lup.break_count,
            lup.streak_stats,
            lup.limit_up_count,
            lup.industry,
            dp.open,
            dp.close,
            dp.high,
            dp.low,
            dp.volume,
            board_strength.board_limit_up_count,
            board_strength.board_pct_chg
        FROM limit_up_pool lup
        LEFT JOIN daily_prices dp
          ON dp.trade_date = lup.trade_date
         AND dp.code = lup.code
        LEFT JOIN board_strength
          ON board_strength.board_name = lup.industry
        WHERE lup.trade_date = ?
        """
        frame = self.store.query(sql, (trade_date, trade_date))
        if frame.empty:
            return frame
        numeric_cols = [
            "latest_price",
            "pct_chg",
            "amount",
            "circ_market_cap",
            "total_market_cap",
            "turnover",
            "seal_amount",
            "break_count",
            "limit_up_count",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "board_limit_up_count",
            "board_pct_chg",
        ]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def compute_emotion_cycle(self, trade_date: str) -> dict:
        history = self.store.query("SELECT * FROM market_metrics ORDER BY trade_date")
        if history.empty:
            detail = {"reason": "缺少 market_metrics 数据"}
            result = pd.DataFrame(
                [{"trade_date": trade_date, "emotion_score": 0, "cycle_label": "未知", "detail_json": json.dumps(detail, ensure_ascii=False)}]
            )
            self.store.upsert_dataframe("emotion_scores", result, unique_columns=["trade_date"])
            return detail | {"emotion_score": 0, "cycle_label": "未知"}

        current = history[history["trade_date"] == trade_date]
        if current.empty:
            raise ValueError(f"trade_date={trade_date} 缺少 market_metrics 记录")
        row = current.iloc[0]
        thresholds = self.config.emotion_thresholds

        def percentile_rank(series: pd.Series, value: float) -> float:
            valid = pd.to_numeric(series, errors="coerce").dropna()
            if valid.empty:
                return 0.0
            return float((valid <= value).mean() * 100)

        limit_up_rank = percentile_rank(history["limit_up_count"], float(row["limit_up_count"]))
        board_height_rank = percentile_rank(history["board_height"], float(row["board_height"]))
        amount_rank = percentile_rank(history["total_amount"], float(row["total_amount"]))
        breadth_value = 0.0
        if float(row["up_count"]) + float(row["down_count"]) > 0:
            breadth_value = float(row["up_count"]) / (float(row["up_count"]) + float(row["down_count"]))
        breadth_score = breadth_value * 100
        streak_score = percentile_rank(history["streak_stock_count"], float(row["streak_stock_count"]))
        score = round(
            limit_up_rank * 0.25
            + board_height_rank * 0.2
            + amount_rank * 0.2
            + breadth_score * 0.2
            + streak_score * 0.15,
            2,
        )
        if score <= thresholds.ice_point_score:
            label = "冰点"
        elif score <= thresholds.repair_score:
            label = "修复"
        elif score <= thresholds.rising_score:
            label = "上升"
        elif score <= thresholds.climax_score:
            label = "分歧"
        else:
            label = "高潮"
        if float(row["limit_down_count"]) > max(float(row["limit_up_count"]) * 0.6, 10) and score >= thresholds.repair_score:
            label = "退潮"
        detail = {
            "emotion_score": score,
            "cycle_label": label,
            "limit_up_rank": round(limit_up_rank, 2),
            "board_height_rank": round(board_height_rank, 2),
            "amount_rank": round(amount_rank, 2),
            "breadth_score": round(breadth_score, 2),
            "streak_score": round(streak_score, 2),
        }
        self.store.upsert_dataframe(
            "emotion_scores",
            pd.DataFrame(
                [
                    {
                        "trade_date": trade_date,
                        "emotion_score": score,
                        "cycle_label": label,
                        "detail_json": json.dumps(detail, ensure_ascii=False),
                    }
                ]
            ),
            unique_columns=["trade_date"],
        )
        return detail

    def _load_stock_character(self, trade_date: str) -> pd.DataFrame:
        begin_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=240)).strftime("%Y%m%d")
        sql = """
        SELECT code,
               COUNT(*) AS limit_up_days,
               SUM(CASE WHEN limit_up_count >= 2 THEN 1 ELSE 0 END) AS streak_days,
               MAX(limit_up_count) AS best_streak
        FROM limit_up_pool
        WHERE trade_date BETWEEN ? AND ?
        GROUP BY code
        """
        return self.store.query(sql, (begin_date, trade_date))

    def _load_recent_returns(self, trade_date: str) -> pd.DataFrame:
        prev_7 = self.store.query(
            """
            WITH ranked AS (
                SELECT trade_date, code, close,
                       ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
                FROM daily_prices
                WHERE trade_date <= ?
            )
            SELECT
                a.code,
                MAX(CASE WHEN a.rn = 1 THEN a.close END) AS close_latest,
                MAX(CASE WHEN a.rn = 7 THEN a.close END) AS close_7d
            FROM ranked a
            GROUP BY a.code
            """,
            (trade_date,),
        )
        if prev_7.empty:
            return prev_7
        prev_7["return_7d"] = ((prev_7["close_latest"] - prev_7["close_7d"]) / prev_7["close_7d"]) * 100
        return prev_7[["code", "return_7d"]]

    def _load_ma_signals(self, trade_date: str) -> pd.DataFrame:
        sql = """
        WITH ranked AS (
            SELECT trade_date, code, close, volume, low,
                   ROW_NUMBER() OVER (PARTITION BY code ORDER BY trade_date DESC) AS rn
            FROM daily_prices
            WHERE trade_date <= ?
        )
        SELECT code,
               AVG(CASE WHEN rn BETWEEN 1 AND 5 THEN close END) AS ma5,
               AVG(CASE WHEN rn BETWEEN 1 AND 10 THEN close END) AS ma10,
               AVG(CASE WHEN rn BETWEEN 1 AND 5 THEN volume END) AS vol5,
               MAX(CASE WHEN rn = 1 THEN close END) AS last_close,
               MAX(CASE WHEN rn = 1 THEN low END) AS last_low,
               MAX(CASE WHEN rn = 1 THEN volume END) AS last_volume
        FROM ranked
        GROUP BY code
        """
        return self.store.query(sql, (trade_date,))

    def _attach_features(self, trade_date: str) -> pd.DataFrame:
        base = self._load_trade_frame(trade_date)
        if base.empty:
            return base
        character = self._load_stock_character(trade_date)
        recent = self._load_recent_returns(trade_date)
        ma_signal = self._load_ma_signals(trade_date)
        merged = base.merge(character, on="code", how="left").merge(recent, on="code", how="left").merge(ma_signal, on="code", how="left")
        merged[["limit_up_days", "streak_days", "best_streak"]] = merged[["limit_up_days", "streak_days", "best_streak"]].fillna(0)
        merged["return_7d"] = merged["return_7d"].fillna(0)
        return merged

    def _tag_board_leader(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["rank_in_industry"] = frame.groupby("industry")["limit_up_count"].rank(method="dense", ascending=False)
        frame["amount_rank_in_industry"] = frame.groupby("industry")["amount"].rank(method="dense", ascending=False)
        frame["is_board_leader"] = (frame["rank_in_industry"] == 1) & (frame["amount_rank_in_industry"] <= 2)
        return frame

    def _apply_hard_filters(self, frame: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        filtered = frame.copy()
        filtered = filtered[filtered["limit_up_count"] < 7]
        filtered = filtered[filtered["return_7d"].fillna(0) < 100]
        if strategy_name == "打板":
            filtered = filtered[filtered["limit_up_count"] >= 2]
        if strategy_name == "2进3":
            filtered = filtered[filtered["limit_up_count"] == 2]
            filtered = filtered[filtered["is_board_leader"]]
        return filtered

    @staticmethod
    def _normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        if numeric.nunique() <= 1:
            return pd.Series([0.5] * len(numeric), index=numeric.index)
        scaled = (numeric - numeric.min()) / (numeric.max() - numeric.min())
        return 1 - scaled if reverse else scaled

    def _score_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        scored = frame.copy()
        board_strength = self._normalize(scored["board_pct_chg"].fillna(0) + scored["board_limit_up_count"].fillna(0))
        stock_character = self._normalize(scored["best_streak"].fillna(0) * 2 + scored["streak_days"].fillna(0))
        turnover = self._normalize(scored["turnover"].fillna(0))
        market_cap = self._normalize(scored["circ_market_cap"].fillna(0), reverse=True)
        streak_height = self._normalize(scored["limit_up_count"].fillna(0))
        theme_space = self._normalize(scored["board_limit_up_count"].fillna(0))
        scored["score"] = (
            board_strength * 30
            + stock_character * 25
            + turnover * 15
            + market_cap * 10
            + streak_height * 10
            + theme_space * 10
        ).round(2)
        scored["tier"] = pd.cut(
            scored["score"],
            bins=[-1, 60, 70, 80, 101],
            labels=["淘汰", "第三梯队", "第二梯队", "第一梯队"],
        ).astype(str)
        return scored[scored["score"] >= 60].sort_values("score", ascending=False)

    def _build_signal_rows(self, trade_date: str, strategy_name: str, frame: pd.DataFrame, cycle_label: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        buy_mode = self.config.buy_rule.default_buy_mode
        signals = frame.copy()
        signals["trade_date"] = trade_date
        signals["strategy_name"] = strategy_name
        signals["buy_mode"] = buy_mode
        signals["buy_price"] = signals["open"].fillna(signals["latest_price"])
        signals["stop_loss"] = signals["buy_price"] * 0.97
        signals["max_position"] = self.config.buy_rule.position_per_stock_limit
        signals["reason"] = (
            "板块="
            + signals["industry"].fillna("未知")
            + "; 连板="
            + signals["limit_up_count"].fillna(0).astype(int).astype(str)
            + "; 板块涨停数="
            + signals["board_limit_up_count"].fillna(0).astype(int).astype(str)
        )
        signals["cycle_label"] = cycle_label
        signals["metrics_json"] = signals.apply(
            lambda row: json.dumps(
                {
                    "turnover": row.get("turnover"),
                    "circ_market_cap": row.get("circ_market_cap"),
                    "board_pct_chg": row.get("board_pct_chg"),
                    "board_limit_up_count": row.get("board_limit_up_count"),
                    "best_streak": row.get("best_streak"),
                },
                ensure_ascii=False,
            ),
            axis=1,
        )
        return signals[
            [
                "trade_date",
                "strategy_name",
                "code",
                "name",
                "score",
                "tier",
                "buy_mode",
                "buy_price",
                "stop_loss",
                "max_position",
                "reason",
                "cycle_label",
                "metrics_json",
            ]
        ]

    def high_relay(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        frame = self._tag_board_leader(self._attach_features(trade_date))
        frame = frame[
            (frame["limit_up_count"].between(3, 6))
            & (frame["circ_market_cap"].between(5e9, 3e10))
            & (frame["turnover"].between(10, 25))
            & (frame["board_limit_up_count"].fillna(0) >= 2)
            & (frame["is_board_leader"])
        ]
        if cycle_label != "上升":
            return pd.DataFrame()
        return self._score_candidates(self._apply_hard_filters(frame, "高位接力"))

    def two_to_three(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        frame = self._tag_board_leader(self._attach_features(trade_date))
        frame = frame[
            (frame["limit_up_count"] == 2)
            & (frame["circ_market_cap"].between(5e9, 2e10))
            & (frame["turnover"].between(5, 20))
            & (frame["is_board_leader"])
        ]
        if cycle_label not in {"上升", "高潮"}:
            return pd.DataFrame()
        return self._score_candidates(self._apply_hard_filters(frame, "2进3"))

    def rebound_after_break(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        prev_date = self._previous_trade_date(trade_date)
        if not prev_date:
            return pd.DataFrame()
        prev_limit = self.store.query(
            "SELECT code, name, limit_up_count, industry FROM limit_up_pool WHERE trade_date = ?",
            (prev_date,),
        )
        today_price = self.store.query(
            "SELECT trade_date, code, name, close, pct_chg, turnover, amount FROM daily_prices WHERE trade_date = ?",
            (trade_date,),
        )
        today_limit = self.store.query("SELECT code FROM limit_up_pool WHERE trade_date = ?", (trade_date,))
        if prev_limit.empty or today_price.empty:
            return pd.DataFrame()
        merged = prev_limit.merge(today_price, on=["code", "name"], how="inner")
        merged = merged[~merged["code"].isin(today_limit["code"].tolist())]
        if merged.empty:
            return merged
        merged = self._tag_board_leader(
            merged.merge(
                self.store.query(
                    "SELECT board_name AS industry, limit_up_count AS board_limit_up_count, pct_chg AS board_pct_chg FROM board_snapshot WHERE trade_date = ?",
                    (trade_date,),
                ),
                on="industry",
                how="left",
            )
        )
        if merged.empty or "is_board_leader" not in merged.columns:
            return pd.DataFrame()
        merged = merged[
            (merged["pct_chg"] < 0)
            & (merged["pct_chg"] > -5)
            & (merged["turnover"] < 15)
            & (merged["is_board_leader"])
        ]
        if cycle_label != "修复":
            return pd.DataFrame()
        merged["circ_market_cap"] = None
        merged["best_streak"] = merged["limit_up_count"]
        merged["streak_days"] = merged["limit_up_count"]
        return self._score_candidates(merged)

    def pullback_dip(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        ma_frame = self._load_ma_signals(trade_date)
        price_frame = self.store.query(
            """
            SELECT trade_date, code, name, open, close, high, low, volume, amount, turnover, pct_chg
            FROM daily_prices
            WHERE trade_date = ?
            """,
            (trade_date,),
        )
        if ma_frame.empty or price_frame.empty:
            return pd.DataFrame()
        merged = price_frame.merge(ma_frame, on="code", how="left")
        merged = merged[
            (merged["ma5"] > merged["ma10"])
            & (merged["last_low"] >= merged["ma10"] * 0.99)
            & (merged["last_close"] <= merged["ma5"] * 1.02)
            & (merged["last_volume"] < merged["vol5"])
        ]
        if merged.empty:
            return merged
        merged["industry"] = None
        merged["board_limit_up_count"] = 0
        merged["board_pct_chg"] = 0
        merged["limit_up_count"] = 0
        merged["best_streak"] = 0
        merged["streak_days"] = 0
        merged["circ_market_cap"] = None
        merged["is_board_leader"] = True
        if cycle_label not in {"冰点", "修复", "退潮"}:
            return pd.DataFrame()
        return self._score_candidates(merged)

    def three_to_four(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        frame = self._tag_board_leader(self._attach_features(trade_date))
        frame = frame[
            (frame["limit_up_count"] == 3)
            & (frame["circ_market_cap"].between(5e9, 2e10))
            & (frame["board_limit_up_count"].fillna(0) >= 2)
        ]
        if cycle_label not in {"上升", "高潮"}:
            return pd.DataFrame()
        return self._score_candidates(self._apply_hard_filters(frame, "3进4"))

    def breakout_limit(self, trade_date: str, cycle_label: str) -> pd.DataFrame:
        frame = self._tag_board_leader(self._attach_features(trade_date))
        frame = frame[(frame["limit_up_count"] >= 2) & (frame["is_board_leader"])]
        if cycle_label != "高潮":
            return pd.DataFrame()
        return self._score_candidates(self._apply_hard_filters(frame, "打板"))

    def run_all_strategies(self, trade_date: str) -> dict[str, pd.DataFrame]:
        emotion = self.compute_emotion_cycle(trade_date)
        cycle_label = emotion["cycle_label"]
        strategy_map = {
            "高位接力": self.high_relay(trade_date, cycle_label),
            "2进3": self.two_to_three(trade_date, cycle_label),
            "断板反包": self.rebound_after_break(trade_date, cycle_label),
            "缩量回踩低吸": self.pullback_dip(trade_date, cycle_label),
            "3进4": self.three_to_four(trade_date, cycle_label),
            "打板": self.breakout_limit(trade_date, cycle_label),
        }
        inserted = 0
        for strategy_name, frame in strategy_map.items():
            signals = self._build_signal_rows(trade_date, strategy_name, frame, cycle_label)
            if signals.empty:
                continue
            inserted += self.store.upsert_dataframe(
                "strategy_signals",
                signals,
                unique_columns=["trade_date", "strategy_name", "code"],
            )
        logger.info("策略信号更新完成 trade_date=%s inserted=%s", trade_date, inserted)
        return strategy_map


if __name__ == "__main__":
    engine = StrategyEngine()
    print(json.dumps(engine.compute_emotion_cycle(datetime.today().strftime("%Y%m%d")), ensure_ascii=False, indent=2))
