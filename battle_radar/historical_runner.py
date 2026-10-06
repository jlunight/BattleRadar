from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date

import akshare as ak
import pandas as pd

from battle_radar.backtest import BacktestEngine
from battle_radar.config import AppConfig, DEFAULT_CONFIG
from battle_radar.data_fetcher import MarketDataFetcher
from battle_radar.database import SQLiteStore
from battle_radar.reporter import ReportBuilder
from battle_radar.strategy import StrategyEngine
from battle_radar.utils import normalize_trade_date


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoricalRunSummary:
    start_date: str
    end_date: str
    trade_days: int
    official_limit_days: int
    derived_limit_days: int
    price_rows: int
    signal_rows: int
    backtest_rows: int
    latest_report_path: str


class HistoricalBackfillRunner:
    def __init__(self, config: AppConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store or SQLiteStore(self.config.db_path)
        self.fetcher = MarketDataFetcher(config=self.config, store=self.store)
        self.strategy = StrategyEngine(config=self.config, store=self.store)
        self.backtest = BacktestEngine(config=self.config, store=self.store)
        self.reporter = ReportBuilder(config=self.config, store=self.store)

    def get_trade_dates(self, start_date: str, end_date: str) -> list[str]:
        start_date = normalize_trade_date(start_date) or start_date
        end_date = normalize_trade_date(end_date) or end_date
        calendar = ak.tool_trade_date_hist_sina()
        calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.strftime("%Y%m%d")
        dates = calendar[(calendar["trade_date"] >= start_date) & (calendar["trade_date"] <= end_date)]["trade_date"].tolist()
        return dates

    def backfill_daily_prices_parallel(
        self,
        universe_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        max_symbols: int | None = None,
        workers: int = 12,
    ) -> int:
        universe = universe_df.head(max_symbols).copy() if max_symbols else universe_df.copy()
        inserted = 0

        def fetch_one(row: pd.Series) -> tuple[str, str, pd.DataFrame | None, str | None]:
            try:
                history = self.fetcher.fetch_daily_history(row["code"], start_date=start_date, end_date=end_date)
                if history.empty:
                    return row["code"], row["name"], None, None
                history["name"] = row["name"]
                return row["code"], row["name"], history, None
            except Exception as exc:  # noqa: BLE001
                return row["code"], row["name"], None, str(exc)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_one, row) for _, row in universe.iterrows()]
            for idx, future in enumerate(as_completed(futures), start=1):
                code, name, history, error = future.result()
                if error:
                    logger.error("并发回补跳过单票 code=%s name=%s error=%s", code, name, error)
                elif history is not None and not history.empty:
                    inserted += self.store.upsert_dataframe(
                        "daily_prices",
                        history,
                        unique_columns=["trade_date", "code"],
                    )
                if idx % 100 == 0 or idx == len(futures):
                    logger.info("历史日线回补进度: %s/%s", idx, len(futures))
        logger.info("历史日线回补完成: %s 条", inserted)
        return inserted

    def _latest_industry_map(self) -> dict[str, str]:
        sql = """
        SELECT code, industry
        FROM limit_up_pool
        WHERE industry IS NOT NULL AND TRIM(industry) <> ''
        ORDER BY trade_date DESC
        """
        df = self.store.query(sql)
        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            mapping.setdefault(row["code"], row["industry"])
        return mapping

    def fetch_official_limit_pools(self, trade_dates: list[str]) -> dict[str, int]:
        official_limit_days = 0
        official_down_days = 0
        for trade_date in trade_dates:
            limit_up = self.fetcher.fetch_limit_up_pool(trade_date)
            limit_down = self.fetcher.fetch_limit_down_pool(trade_date)
            if not limit_up.empty:
                official_limit_days += 1
                self.store.upsert_dataframe("limit_up_pool", limit_up, unique_columns=["trade_date", "code"])
            if not limit_down.empty:
                official_down_days += 1
                self.store.upsert_dataframe("limit_down_pool", limit_down, unique_columns=["trade_date", "code"])
        return {"official_limit_days": official_limit_days, "official_down_days": official_down_days}

    def derive_limit_pools(self, start_date: str, end_date: str, official_trade_dates: set[str]) -> dict[str, int]:
        start_date = normalize_trade_date(start_date) or start_date
        end_date = normalize_trade_date(end_date) or end_date
        price_df = self.store.query(
            """
            SELECT trade_date, code, name, open, close, high, low, volume, amount, turnover, pct_chg
            FROM daily_prices
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY code, trade_date
            """,
            (start_date, end_date),
        )
        if price_df.empty:
            return {"derived_limit_days": 0, "derived_limit_rows": 0, "derived_down_rows": 0}

        price_df["name"] = price_df["name"].fillna("")
        price_df = price_df[~price_df["name"].str.contains("ST", case=False, na=False)].copy()
        price_df["turnover"] = pd.to_numeric(price_df["turnover"], errors="coerce")
        price_df["amount"] = pd.to_numeric(price_df["amount"], errors="coerce")
        price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
        price_df["high"] = pd.to_numeric(price_df["high"], errors="coerce")
        price_df["low"] = pd.to_numeric(price_df["low"], errors="coerce")
        price_df["pct_chg"] = pd.to_numeric(price_df["pct_chg"], errors="coerce")
        price_df["circ_market_cap"] = price_df.apply(
            lambda row: (row["amount"] / (row["turnover"] / 100.0)) if pd.notna(row["turnover"]) and row["turnover"] > 0 else None,
            axis=1,
        )
        price_df["is_limit_up"] = (price_df["pct_chg"] >= 9.7) & (price_df["close"] >= price_df["high"] * 0.999)
        price_df["is_limit_down"] = (price_df["pct_chg"] <= -9.7) & (price_df["close"] <= price_df["low"] * 1.001)

        streak_values: list[int] = []
        for _, group in price_df.groupby("code", sort=False):
            streak = 0
            for flag in group["is_limit_up"].tolist():
                streak = streak + 1 if flag else 0
                streak_values.append(streak)
        price_df["derived_streak"] = streak_values

        industry_map = self._latest_industry_map()
        derived_dates = sorted(set(price_df.loc[price_df["is_limit_up"], "trade_date"].tolist()) - set(official_trade_dates))

        derived_up_rows: list[dict] = []
        derived_down_rows: list[dict] = []
        for _, row in price_df.iterrows():
            trade_date = row["trade_date"]
            if trade_date in official_trade_dates:
                continue
            if row["is_limit_up"]:
                streak = int(row["derived_streak"])
                derived_up_rows.append(
                    {
                        "trade_date": trade_date,
                        "code": row["code"],
                        "name": row["name"],
                        "latest_price": row["close"],
                        "pct_chg": row["pct_chg"],
                        "amount": row["amount"],
                        "circ_market_cap": row["circ_market_cap"],
                        "total_market_cap": row["circ_market_cap"],
                        "turnover": row["turnover"],
                        "seal_amount": None,
                        "first_limit_time": None,
                        "last_limit_time": None,
                        "break_count": 0,
                        "streak_stats": json.dumps({"days": streak, "ct": streak, "source": "derived"}, ensure_ascii=False),
                        "limit_up_count": streak,
                        "industry": industry_map.get(row["code"]),
                        "limit_up_reason": "derived_from_daily_prices",
                    }
                )
            if row["is_limit_down"]:
                derived_down_rows.append(
                    {
                        "trade_date": trade_date,
                        "code": row["code"],
                        "name": row["name"],
                        "latest_price": row["close"],
                        "pct_chg": row["pct_chg"],
                        "amount": row["amount"],
                        "circ_market_cap": row["circ_market_cap"],
                        "total_market_cap": row["circ_market_cap"],
                        "pe_dynamic": None,
                        "turnover": row["turnover"],
                        "seal_amount": None,
                        "last_limit_time": None,
                        "board_amount": None,
                        "consecutive_limit_down": 1,
                        "break_count": 0,
                        "industry": industry_map.get(row["code"]),
                    }
                )
        if derived_up_rows:
            self.store.upsert_dataframe(
                "limit_up_pool",
                pd.DataFrame(derived_up_rows),
                unique_columns=["trade_date", "code"],
            )
        if derived_down_rows:
            self.store.upsert_dataframe(
                "limit_down_pool",
                pd.DataFrame(derived_down_rows),
                unique_columns=["trade_date", "code"],
            )
        return {
            "derived_limit_days": len(derived_dates),
            "derived_limit_rows": len(derived_up_rows),
            "derived_down_rows": len(derived_down_rows),
        }

    def recompute_market_metrics_for_dates(self, trade_dates: list[str]) -> None:
        for trade_date in trade_dates:
            self.fetcher.compute_market_metrics(trade_date)

    def run_strategies_for_dates(self, trade_dates: list[str]) -> int:
        total = 0
        for trade_date in trade_dates:
            strategy_map = self.strategy.run_all_strategies(trade_date)
            total += sum(len(frame) for frame in strategy_map.values())
        return total

    def run(
        self,
        start_date: str,
        end_date: str | None = None,
        max_symbols: int | None = None,
    ) -> HistoricalRunSummary:
        start_date = normalize_trade_date(start_date) or start_date
        end_date = normalize_trade_date(end_date or date.today().strftime("%Y%m%d")) or date.today().strftime("%Y%m%d")
        max_symbols = max_symbols if max_symbols and max_symbols > 0 else None
        trade_dates = self.get_trade_dates(start_date, end_date)
        if not trade_dates:
            raise ValueError(f"区间内无交易日: {start_date} - {end_date}")

        self.fetcher.normalize_stored_trade_dates()
        universe = self.fetcher.update_stock_universe()
        if universe.empty:
            universe = self.store.query("SELECT code, name, exchange, board FROM stock_universe ORDER BY code")
            logger.warning("实时股票池获取为空，回退使用库内股票池: %s", len(universe))
        price_rows = self.backfill_daily_prices_parallel(
            universe_df=universe,
            start_date=start_date,
            end_date=end_date,
            max_symbols=max_symbols,
        )
        official_stats = self.fetch_official_limit_pools(trade_dates)
        derived_stats = self.derive_limit_pools(start_date, end_date, official_trade_dates=set(
            self.store.query(
                "SELECT DISTINCT trade_date FROM limit_up_pool WHERE trade_date BETWEEN ? AND ?",
                (start_date, end_date),
            )["trade_date"].tolist()
        ))
        self.recompute_market_metrics_for_dates(trade_dates)
        signal_rows = self.run_strategies_for_dates(trade_dates)
        backtest_rows = len(self.backtest.backtest_signals())
        latest_report_path = str(self.reporter.write_report(trade_dates[-1]))
        return HistoricalRunSummary(
            start_date=start_date,
            end_date=end_date,
            trade_days=len(trade_dates),
            official_limit_days=official_stats["official_limit_days"],
            derived_limit_days=derived_stats["derived_limit_days"],
            price_rows=price_rows,
            signal_rows=signal_rows,
            backtest_rows=backtest_rows,
            latest_report_path=latest_report_path,
        )
