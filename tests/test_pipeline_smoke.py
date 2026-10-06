from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from battle_radar.backtest import BacktestEngine
from battle_radar.config import AppConfig
from battle_radar.database import SQLiteStore
from battle_radar.reporter import ReportBuilder
from battle_radar.strategy import StrategyEngine


class PipelineSmokeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config = AppConfig(
            workspace_dir=root,
            data_dir=root,
            reports_dir=root / "reports",
            db_path=root / "battle_radar.db",
        )
        self.store = SQLiteStore(self.config.db_path)
        self.strategy = StrategyEngine(config=self.config, store=self.store)
        self.backtest = BacktestEngine(config=self.config, store=self.store)
        self.reporter = ReportBuilder(config=self.config, store=self.store)
        self._seed_data()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _seed_data(self) -> None:
        daily_prices = pd.DataFrame(
            [
                {"trade_date": "20261001", "code": "002058", "name": "紫竹高科", "open": 10.0, "close": 11.0, "high": 11.0, "low": 9.9, "volume": 1000, "amount": 1000000, "turnover": 12.0, "amplitude": 11.0, "pct_chg": 10.0, "chg": 1.0, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261001", "code": "600000", "name": "浦发银行", "open": 10.0, "close": 10.1, "high": 10.2, "low": 9.8, "volume": 1000, "amount": 1000000, "turnover": 2.0, "amplitude": 4.0, "pct_chg": 1.0, "chg": 0.1, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261002", "code": "002058", "name": "紫竹高科", "open": 11.0, "close": 12.1, "high": 12.1, "low": 10.9, "volume": 1200, "amount": 1200000, "turnover": 13.0, "amplitude": 10.0, "pct_chg": 10.0, "chg": 1.1, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261002", "code": "600000", "name": "浦发银行", "open": 10.1, "close": 10.2, "high": 10.25, "low": 10.0, "volume": 1000, "amount": 1000000, "turnover": 2.1, "amplitude": 2.0, "pct_chg": 1.0, "chg": 0.1, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261003", "code": "002058", "name": "紫竹高科", "open": 12.0, "close": 13.31, "high": 13.31, "low": 11.95, "volume": 1400, "amount": 1400000, "turnover": 14.0, "amplitude": 11.0, "pct_chg": 10.0, "chg": 1.21, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261003", "code": "600000", "name": "浦发银行", "open": 10.2, "close": 10.15, "high": 10.3, "low": 10.1, "volume": 1000, "amount": 1000000, "turnover": 2.0, "amplitude": 1.9, "pct_chg": -0.5, "chg": -0.05, "market_cap": None, "circ_market_cap": 10000000000},
                {"trade_date": "20261004", "code": "002058", "name": "紫竹高科", "open": 13.4, "close": 13.8, "high": 14.2, "low": 13.2, "volume": 1500, "amount": 1500000, "turnover": 15.0, "amplitude": 8.0, "pct_chg": 3.68, "chg": 0.49, "market_cap": None, "circ_market_cap": 10000000000},
            ]
        )
        self.store.upsert_dataframe("daily_prices", daily_prices, unique_columns=["trade_date", "code"])

        limit_up_pool = pd.DataFrame(
            [
                {"trade_date": "20261001", "code": "002058", "name": "紫竹高科", "latest_price": 11.0, "pct_chg": 10.0, "amount": 1000000, "circ_market_cap": 10000000000, "total_market_cap": 12000000000, "turnover": 12.0, "seal_amount": 10000, "first_limit_time": "093000", "last_limit_time": "145700", "break_count": 0, "streak_stats": '{"days":1,"ct":1}', "limit_up_count": 1, "industry": "电池", "limit_up_reason": None},
                {"trade_date": "20261002", "code": "002058", "name": "紫竹高科", "latest_price": 12.1, "pct_chg": 10.0, "amount": 1200000, "circ_market_cap": 10000000000, "total_market_cap": 12000000000, "turnover": 13.0, "seal_amount": 12000, "first_limit_time": "093000", "last_limit_time": "145700", "break_count": 0, "streak_stats": '{"days":2,"ct":2}', "limit_up_count": 2, "industry": "电池", "limit_up_reason": None},
                {"trade_date": "20261003", "code": "002058", "name": "紫竹高科", "latest_price": 13.31, "pct_chg": 10.0, "amount": 1400000, "circ_market_cap": 10000000000, "total_market_cap": 12000000000, "turnover": 14.0, "seal_amount": 14000, "first_limit_time": "093000", "last_limit_time": "145700", "break_count": 0, "streak_stats": '{"days":3,"ct":3}', "limit_up_count": 3, "industry": "电池", "limit_up_reason": None},
            ]
        )
        self.store.upsert_dataframe("limit_up_pool", limit_up_pool, unique_columns=["trade_date", "code"])

        board_snapshot = pd.DataFrame(
            [
                {"trade_date": "20261003", "board_type": "industry", "board_code": "BK0001", "board_name": "电池", "latest_price": 100, "pct_chg": 5.5, "chg": 5.0, "total_market_cap": 100000000000, "turnover": 6.0, "up_count": 30, "down_count": 2, "leader_stock": "紫竹高科", "leader_pct_chg": 10.0, "limit_up_count": 3},
                {"trade_date": "20261003", "board_type": "concept", "board_code": "BK9999", "board_name": "储能", "latest_price": 95, "pct_chg": 4.0, "chg": 3.8, "total_market_cap": 90000000000, "turnover": 5.5, "up_count": 20, "down_count": 5, "leader_stock": "紫竹高科", "leader_pct_chg": 10.0, "limit_up_count": 2},
            ]
        )
        self.store.upsert_dataframe(
            "board_snapshot",
            board_snapshot,
            unique_columns=["trade_date", "board_type", "board_code"],
        )

        metrics = pd.DataFrame(
            [
                {"trade_date": "20261001", "up_count": 3000, "down_count": 1500, "flat_count": 100, "limit_up_count": 50, "limit_down_count": 3, "total_amount": 900000000000, "board_height": 1, "streak_stock_count": 10, "source": "seed"},
                {"trade_date": "20261002", "up_count": 3200, "down_count": 1300, "flat_count": 100, "limit_up_count": 70, "limit_down_count": 2, "total_amount": 1000000000000, "board_height": 2, "streak_stock_count": 16, "source": "seed"},
                {"trade_date": "20261003", "up_count": 3600, "down_count": 900, "flat_count": 100, "limit_up_count": 90, "limit_down_count": 1, "total_amount": 1100000000000, "board_height": 3, "streak_stock_count": 25, "source": "seed"},
            ]
        )
        self.store.upsert_dataframe("market_metrics", metrics, unique_columns=["trade_date"])

    def test_strategy_backtest_report_smoke(self) -> None:
        emotion = self.strategy.compute_emotion_cycle("20261003")
        self.assertIn("cycle_label", emotion)
        result = self.strategy.run_all_strategies("20261003")
        self.assertTrue(any(len(df) > 0 for df in result.values()))

        signals = self.store.query("SELECT * FROM strategy_signals WHERE trade_date = '20261003'")
        self.assertGreater(len(signals), 0)

        backtest_result = self.backtest.backtest_signals(signal_date="20261003")
        self.assertGreater(len(backtest_result), 0)
        self.assertTrue((backtest_result["max_return_pct"] >= 4).any())

        report_path = self.reporter.write_report("20261003")
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("市场情绪总览", content)
        self.assertIn("战法胜率总排名", content)


if __name__ == "__main__":
    unittest.main()
