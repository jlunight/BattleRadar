from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from battle_radar.backtest import BacktestEngine
from battle_radar.config import AppConfig
from battle_radar.database import SQLiteStore


class BacktestSummaryTestCase(unittest.TestCase):
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
        self.engine = BacktestEngine(config=self.config, store=self.store)
        self._seed_data()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _seed_data(self) -> None:
        strategy_signals = pd.DataFrame(
            [
                {"trade_date": "20261001", "strategy_name": "2进3", "code": "000001", "name": "样本A1", "cycle_label": "上升"},
                {"trade_date": "20261002", "strategy_name": "2进3", "code": "000002", "name": "样本A2", "cycle_label": "上升"},
                {"trade_date": "20261003", "strategy_name": "2进3", "code": "000003", "name": "样本A3", "cycle_label": "上升"},
                {"trade_date": "20261004", "strategy_name": "2进3", "code": "000004", "name": "样本A4", "cycle_label": "上升"},
                {"trade_date": "20261005", "strategy_name": "2进3", "code": "000005", "name": "样本A5", "cycle_label": "上升"},
                {"trade_date": "20261006", "strategy_name": "2进3", "code": "000006", "name": "样本A6", "cycle_label": "上升"},
                {"trade_date": "20261007", "strategy_name": "2进3", "code": "000007", "name": "样本A7", "cycle_label": "上升"},
                {"trade_date": "20261008", "strategy_name": "2进3", "code": "000008", "name": "样本A8", "cycle_label": "修复"},
                {"trade_date": "20261001", "strategy_name": "打板", "code": "000101", "name": "样本B1", "cycle_label": "修复"},
                {"trade_date": "20261002", "strategy_name": "打板", "code": "000102", "name": "样本B2", "cycle_label": "修复"},
                {"trade_date": "20261003", "strategy_name": "打板", "code": "000103", "name": "样本B3", "cycle_label": "修复"},
                {"trade_date": "20261004", "strategy_name": "打板", "code": "000104", "name": "样本B4", "cycle_label": "退潮"},
                {"trade_date": "20261001", "strategy_name": "高位接力", "code": "000201", "name": "样本C1", "cycle_label": "上升"},
                {"trade_date": "20261002", "strategy_name": "高位接力", "code": "000202", "name": "样本C2", "cycle_label": "上升"},
                {"trade_date": "20261003", "strategy_name": "高位接力", "code": "000203", "name": "样本C3", "cycle_label": "上升"},
                {"trade_date": "20261004", "strategy_name": "高位接力", "code": "000204", "name": "样本C4", "cycle_label": "上升"},
                {"trade_date": "20261005", "strategy_name": "高位接力", "code": "000205", "name": "样本C5", "cycle_label": "上升"},
                {"trade_date": "20261006", "strategy_name": "高位接力", "code": "000206", "name": "样本C6", "cycle_label": "上升"},
                {"trade_date": "20261007", "strategy_name": "高位接力", "code": "000207", "name": "样本C7", "cycle_label": "修复"},
                {"trade_date": "20261008", "strategy_name": "高位接力", "code": "000208", "name": "样本C8", "cycle_label": "修复"},
            ]
        )
        self.store.upsert_dataframe(
            "strategy_signals",
            strategy_signals,
            unique_columns=["trade_date", "strategy_name", "code"],
        )

        backtest_results = pd.DataFrame(
            [
                {"signal_date": "20261001", "strategy_name": "2进3", "code": "000001", "success": 1, "max_return_pct": 7.0, "pnl_ratio": 4.0},
                {"signal_date": "20261002", "strategy_name": "2进3", "code": "000002", "success": 1, "max_return_pct": 6.2, "pnl_ratio": 4.0},
                {"signal_date": "20261003", "strategy_name": "2进3", "code": "000003", "success": 1, "max_return_pct": 5.5, "pnl_ratio": 4.0},
                {"signal_date": "20261004", "strategy_name": "2进3", "code": "000004", "success": 1, "max_return_pct": 4.8, "pnl_ratio": 4.0},
                {"signal_date": "20261005", "strategy_name": "2进3", "code": "000005", "success": 1, "max_return_pct": 6.0, "pnl_ratio": 4.0},
                {"signal_date": "20261006", "strategy_name": "2进3", "code": "000006", "success": 1, "max_return_pct": 4.3, "pnl_ratio": 4.0},
                {"signal_date": "20261007", "strategy_name": "2进3", "code": "000007", "success": 1, "max_return_pct": 4.5, "pnl_ratio": 4.0},
                {"signal_date": "20261008", "strategy_name": "2进3", "code": "000008", "success": 0, "max_return_pct": 1.2, "pnl_ratio": -2.5},
                {"signal_date": "20261001", "strategy_name": "打板", "code": "000101", "success": 1, "max_return_pct": 5.0, "pnl_ratio": 4.0},
                {"signal_date": "20261002", "strategy_name": "打板", "code": "000102", "success": 0, "max_return_pct": 1.0, "pnl_ratio": -2.0},
                {"signal_date": "20261003", "strategy_name": "打板", "code": "000103", "success": 1, "max_return_pct": 4.5, "pnl_ratio": 4.0},
                {"signal_date": "20261004", "strategy_name": "打板", "code": "000104", "success": 0, "max_return_pct": -1.0, "pnl_ratio": -3.0},
                {"signal_date": "20261001", "strategy_name": "高位接力", "code": "000201", "success": 0, "max_return_pct": 1.5, "pnl_ratio": -2.0},
                {"signal_date": "20261002", "strategy_name": "高位接力", "code": "000202", "success": 0, "max_return_pct": 0.8, "pnl_ratio": -3.1},
                {"signal_date": "20261003", "strategy_name": "高位接力", "code": "000203", "success": 0, "max_return_pct": -1.2, "pnl_ratio": -4.0},
                {"signal_date": "20261004", "strategy_name": "高位接力", "code": "000204", "success": 1, "max_return_pct": 4.3, "pnl_ratio": 4.0},
                {"signal_date": "20261005", "strategy_name": "高位接力", "code": "000205", "success": 0, "max_return_pct": 1.7, "pnl_ratio": -2.6},
                {"signal_date": "20261006", "strategy_name": "高位接力", "code": "000206", "success": 0, "max_return_pct": 2.1, "pnl_ratio": -1.4},
                {"signal_date": "20261007", "strategy_name": "高位接力", "code": "000207", "success": 0, "max_return_pct": 0.3, "pnl_ratio": -3.0},
                {"signal_date": "20261008", "strategy_name": "高位接力", "code": "000208", "success": 0, "max_return_pct": -2.0, "pnl_ratio": -5.0},
            ]
        )
        self.store.upsert_dataframe(
            "backtest_results",
            backtest_results,
            unique_columns=["signal_date", "strategy_name", "code"],
        )

    def test_summary_produces_strategy_diagnostics(self) -> None:
        summary = self.engine.summarize_performance()
        self.assertListEqual(
            summary["strategy_name"].tolist(),
            ["2进3", "打板", "高位接力"],
        )

        enabled = summary.loc[summary["strategy_name"] == "2进3"].iloc[0]
        self.assertEqual(enabled["recommended_status"], "启用")
        self.assertEqual(enabled["sample_level"], "样本充足")
        self.assertEqual(enabled["best_cycle_label"], "上升")
        self.assertEqual(enabled["sample_count"], 8)
        self.assertEqual(enabled["bayesian_win_rate"], 80.0)

        observe = summary.loc[summary["strategy_name"] == "打板"].iloc[0]
        self.assertEqual(observe["recommended_status"], "观察")
        self.assertIn(observe["sample_level"], {"样本偏少", "观察期"})

        disabled = summary.loc[summary["strategy_name"] == "高位接力"].iloc[0]
        self.assertEqual(disabled["recommended_status"], "停用")
        self.assertLess(disabled["bayesian_win_rate"], 40.0)


if __name__ == "__main__":
    unittest.main()
