from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from battle_radar.backtest import BacktestEngine
from battle_radar.config import AppConfig, DEFAULT_CONFIG
from battle_radar.database import SQLiteStore


class ReportBuilder:
    def __init__(self, config: AppConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store or SQLiteStore(self.config.db_path)
        self.backtest = BacktestEngine(config=self.config, store=self.store)

    def _previous_trade_date(self, trade_date: str) -> str | None:
        dates = self.store.query("SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date")["trade_date"].tolist()
        if trade_date not in dates:
            earlier = [d for d in dates if d < trade_date]
            return earlier[-1] if earlier else None
        idx = dates.index(trade_date)
        return dates[idx - 1] if idx > 0 else None

    @staticmethod
    def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "暂无数据\n"
        subset = frame[columns].copy()
        return subset.to_markdown(index=False) + "\n"

    def build_markdown(self, trade_date: str) -> str:
        prev_date = self._previous_trade_date(trade_date)
        emotion = self.store.query("SELECT * FROM emotion_scores WHERE trade_date = ?", (trade_date,))
        prev_emotion = self.store.query("SELECT * FROM emotion_scores WHERE trade_date = ?", (prev_date,)) if prev_date else pd.DataFrame()
        metrics = self.store.query("SELECT * FROM market_metrics WHERE trade_date = ?", (trade_date,))
        ladder = self.store.query(
            """
            SELECT code, name, limit_up_count, turnover, industry, limit_up_reason
            FROM limit_up_pool
            WHERE trade_date = ?
            ORDER BY limit_up_count DESC, turnover DESC
            """,
            (trade_date,),
        )
        broken = self.store.query(
            """
            SELECT dp.code, dp.name, dp.pct_chg, dp.turnover
            FROM daily_prices dp
            INNER JOIN limit_up_pool prev
              ON prev.code = dp.code
             AND prev.trade_date = ?
            LEFT JOIN limit_up_pool curr
              ON curr.code = dp.code
             AND curr.trade_date = ?
            WHERE dp.trade_date = ?
              AND curr.code IS NULL
            ORDER BY dp.pct_chg DESC
            LIMIT 20
            """,
            (prev_date, trade_date, trade_date),
        ) if prev_date else pd.DataFrame()
        validation = self.backtest.latest_validation(prev_date) if prev_date else pd.DataFrame()
        ranking = self.backtest.summarize_performance()
        picks = self.store.query(
            """
            SELECT strategy_name, code, name, score, tier, buy_price, stop_loss, max_position, reason
            FROM strategy_signals
            WHERE trade_date = ?
            ORDER BY score DESC, strategy_name, code
            """,
            (trade_date,),
        )
        emotion_row = emotion.iloc[0] if not emotion.empty else {}
        prev_row = prev_emotion.iloc[0] if not prev_emotion.empty else {}
        metrics_row = metrics.iloc[0] if not metrics.empty else {}
        emotion_delta = None
        if emotion_row is not None and isinstance(emotion_row, pd.Series) and not prev_emotion.empty:
            emotion_delta = round(float(emotion_row["emotion_score"]) - float(prev_row["emotion_score"]), 2)

        lines: list[str] = []
        lines.append(f"# A股短线战法复盘报告 - {trade_date}")
        lines.append("")
        lines.append("## 1. 市场情绪总览")
        lines.append(
            f"- 情绪温度：{emotion_row.get('emotion_score', 'N/A')} / 100；周期定位：{emotion_row.get('cycle_label', '未知')}；较前一日变化：{emotion_delta if emotion_delta is not None else 'N/A'}"
        )
        lines.append(
            f"- 涨停数：{metrics_row.get('limit_up_count', 'N/A')}；跌停数：{metrics_row.get('limit_down_count', 'N/A')}；上涨家数：{metrics_row.get('up_count', 'N/A')}；下跌家数：{metrics_row.get('down_count', 'N/A')}；两市成交额：{metrics_row.get('total_amount', 'N/A')}"
        )
        lines.append("")
        lines.append("## 2. 连板梯队")
        if ladder.empty:
            lines.append("暂无连板数据")
        else:
            ladder_view = ladder.copy()
            ladder_view["异动预警"] = ladder_view["limit_up_count"].map(lambda x: "高位预警" if float(x) >= 6 else "")
            lines.append(
                self._format_table(
                    ladder_view.head(30),
                    ["code", "name", "limit_up_count", "turnover", "industry", "limit_up_reason", "异动预警"],
                )
            )
        lines.append("### 断板股")
        lines.append(self._format_table(broken, ["code", "name", "pct_chg", "turnover"]))
        lines.append("## 3. 昨日战法实战验证")
        lines.append(
            self._format_table(
                validation,
                ["strategy_name", "code", "name", "buy_price", "next_high", "max_return_pct", "success", "sample_tag"],
            )
        )
        lines.append("## 4. 战法胜率总排名")
        lines.append(
            self._format_table(
                ranking,
                ["strategy_name", "sample_count", "cumulative_win_rate", "profit_loss_ratio", "rolling_30d_win_rate"],
            )
        )
        lines.append("## 5. 次日战法选择与标的优先级")
        lines.append(
            self._format_table(
                picks,
                ["strategy_name", "tier", "code", "name", "score", "buy_price", "stop_loss", "max_position", "reason"],
            )
        )
        lines.append("## 6. 操作时间表")
        lines.append("- 09:15-09:25：检查隔夜消息、核对情绪周期、排除异动与停牌风险。")
        lines.append("- 09:25-09:35：观察竞价强弱，仅保留量化规则前列个股。")
        lines.append("- 09:35-10:30：执行主攻战法；高位接力与 2进3 仅在主线加强时参与。")
        lines.append("- 10:30-14:00：低吸策略等回踩确认；不做追涨情绪单。")
        lines.append("- 14:00-15:00：记录候选股分时表现，收盘后更新验证与报告。")
        lines.append("## 7. 核心结论")
        top_pick = picks.iloc[0] if not picks.empty else {}
        lines.append(f"- 情绪阶段为 {emotion_row.get('cycle_label', '未知')}，仓位上限遵守单只≤30%、总仓≤80%。")
        lines.append("- 所有标的仅由规则引擎产生，禁止临盘情绪化加仓或切换战法。")
        lines.append("- 高位接力统一执行 +4% 止盈；若交易失败则无条件止损。")
        lines.append(f"- 当前优先级最高标的：{top_pick.get('code', '暂无')} {top_pick.get('name', '')}。")
        lines.append("- 报告基于 SQLite 中累计样本自动生成，历史轨迹只追加不覆盖。")
        return "\n".join(lines).strip() + "\n"

    def write_report(self, trade_date: str) -> Path:
        content = self.build_markdown(trade_date)
        output = self.config.reports_dir / f"report_{trade_date}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        return output


if __name__ == "__main__":
    builder = ReportBuilder()
    report_path = builder.write_report(datetime.today().strftime("%Y%m%d"))
    print(report_path)
