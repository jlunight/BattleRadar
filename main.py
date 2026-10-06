from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from battle_radar.backtest import BacktestEngine
from battle_radar.config import DEFAULT_CONFIG
from battle_radar.data_fetcher import MarketDataFetcher, configure_logging
from battle_radar.reporter import ReportBuilder
from battle_radar.strategy import StrategyEngine


logger = logging.getLogger(__name__)


def run_data_layer(args: argparse.Namespace) -> dict:
    fetcher = MarketDataFetcher(config=DEFAULT_CONFIG)
    return fetcher.update_data_layer(
        trade_date=args.trade_date,
        start_date=args.start_date,
        max_symbols=args.max_symbols,
        update_board_members=not args.skip_board_members,
    )


def run_strategy_layer(args: argparse.Namespace) -> dict:
    engine = StrategyEngine(config=DEFAULT_CONFIG)
    emotion = engine.compute_emotion_cycle(args.trade_date)
    strategies = engine.run_all_strategies(args.trade_date)
    return {
        "trade_date": args.trade_date,
        "emotion": emotion,
        "strategy_counts": {name: len(frame) for name, frame in strategies.items()},
    }


def run_backtest_layer(args: argparse.Namespace) -> dict:
    engine = BacktestEngine(config=DEFAULT_CONFIG)
    result = engine.backtest_signals(signal_date=args.signal_date)
    summary = engine.summarize_performance()
    return {
        "backtest_rows": len(result),
        "summary": summary.to_dict(orient="records"),
    }


def run_report_layer(args: argparse.Namespace) -> dict:
    builder = ReportBuilder(config=DEFAULT_CONFIG)
    report_path = builder.write_report(args.trade_date)
    return {"report_path": str(report_path)}


def run_all(args: argparse.Namespace) -> dict:
    data_result = run_data_layer(args)
    strategy_result = run_strategy_layer(args)
    backtest_result = run_backtest_layer(args)
    report_result = run_report_layer(args)
    return {
        "data_layer": data_result,
        "strategy_layer": strategy_result,
        "backtest_layer": backtest_result,
        "report_layer": report_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BattleRadar A股短线战法量化分析与回测系统")
    parser.add_argument(
        "--phase",
        choices=["data", "strategy", "backtest", "report", "all"],
        default="all",
        help="执行阶段",
    )
    parser.add_argument("--trade-date", default=date.today().strftime("%Y%m%d"), help="交易日 YYYYMMDD")
    parser.add_argument("--start-date", default=DEFAULT_CONFIG.backfill_start_date, help="回补起始日期 YYYYMMDD")
    parser.add_argument("--signal-date", default=None, help="仅回测指定信号日期")
    parser.add_argument("--max-symbols", type=int, default=60, help="数据层测试时限制股票数量")
    parser.add_argument("--skip-board-members", action="store_true", help="跳过板块成分更新")
    return parser


def main() -> None:
    configure_logging(DEFAULT_CONFIG.log_level)
    parser = build_parser()
    args = parser.parse_args()
    runner = {
        "data": run_data_layer,
        "strategy": run_strategy_layer,
        "backtest": run_backtest_layer,
        "report": run_report_layer,
        "all": run_all,
    }[args.phase]
    result = runner(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
