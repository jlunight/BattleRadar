from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from battle_radar.config import AppConfig
from battle_radar.data_fetcher import MarketDataFetcher
from battle_radar.database import SQLiteStore


class DataLayerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test.db"
        config = AppConfig(
            workspace_dir=Path(self.temp_dir.name),
            data_dir=Path(self.temp_dir.name),
            reports_dir=Path(self.temp_dir.name),
            db_path=db_path,
        )
        self.store = SQLiteStore(db_path)
        self.fetcher = MarketDataFetcher(config=config, store=self.store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_main_board_filter(self) -> None:
        codes = ["600000", "000001", "002594", "003816", "300750", "688981", "920001"]
        result = [code for code in codes if self.fetcher.is_main_board_code(code)]
        self.assertEqual(result, ["600000", "000001", "002594", "003816"])

    @patch.object(MarketDataFetcher, "_fetch_paginated_clist")
    def test_update_stock_universe(self, mocked_fetch: patch) -> None:
        mocked_fetch.return_value = pd.DataFrame(
            [
                {"f12": "600000", "f14": "浦发银行", "f13": 1},
                {"f12": "300750", "f14": "宁德时代", "f13": 0},
                {"f12": "002594", "f14": "比亚迪", "f13": 0},
            ]
        )
        universe = self.fetcher.update_stock_universe()
        self.assertEqual(universe["code"].tolist(), ["002594", "600000"])
        stored = self.store.query("SELECT code, name FROM stock_universe ORDER BY code")
        self.assertEqual(len(stored), 2)

    def test_compute_market_metrics(self) -> None:
        self.store.upsert_dataframe(
            "daily_prices",
            pd.DataFrame(
                [
                    {"trade_date": "20261005", "code": "600000", "name": "浦发银行", "open": 10, "close": 11, "high": 11.2, "low": 9.8, "volume": 1000, "amount": 10000, "turnover": 1, "amplitude": 1, "pct_chg": 5, "chg": 0.5, "market_cap": None, "circ_market_cap": None},
                    {"trade_date": "20261005", "code": "002594", "name": "比亚迪", "open": 20, "close": 19, "high": 20.2, "low": 18.8, "volume": 1000, "amount": 20000, "turnover": 1, "amplitude": 1, "pct_chg": -5, "chg": -1, "market_cap": None, "circ_market_cap": None},
                ]
            ),
            unique_columns=["trade_date", "code"],
        )
        self.store.upsert_dataframe(
            "limit_up_pool",
            pd.DataFrame(
                [
                    {
                        "trade_date": "20261005",
                        "code": "600000",
                        "name": "浦发银行",
                        "latest_price": 11,
                        "pct_chg": 10,
                        "amount": 10000,
                        "circ_market_cap": 10000000000,
                        "total_market_cap": 12000000000,
                        "turnover": 10,
                        "seal_amount": 1000,
                        "first_limit_time": "093000",
                        "last_limit_time": "145900",
                        "break_count": 0,
                        "streak_stats": '{"days":1,"ct":1}',
                        "limit_up_count": 2,
                        "industry": "银行",
                        "limit_up_reason": None,
                    }
                ]
            ),
            unique_columns=["trade_date", "code"],
        )
        self.store.upsert_dataframe(
            "limit_down_pool",
            pd.DataFrame(
                [
                    {
                        "trade_date": "20261005",
                        "code": "002594",
                        "name": "比亚迪",
                        "latest_price": 19,
                        "pct_chg": -10,
                        "amount": 20000,
                        "circ_market_cap": 10000000000,
                        "total_market_cap": 12000000000,
                        "pe_dynamic": 20,
                        "turnover": 10,
                        "seal_amount": 1000,
                        "last_limit_time": "145900",
                        "board_amount": 1000,
                        "consecutive_limit_down": 1,
                        "break_count": 0,
                        "industry": "新能源",
                    }
                ]
            ),
            unique_columns=["trade_date", "code"],
        )
        metrics = self.fetcher.compute_market_metrics("20261005")
        self.assertEqual(int(metrics.iloc[0]["limit_up_count"]), 1)
        self.assertEqual(int(metrics.iloc[0]["limit_down_count"]), 1)
        self.assertEqual(int(metrics.iloc[0]["board_height"]), 2)


if __name__ == "__main__":
    unittest.main()
