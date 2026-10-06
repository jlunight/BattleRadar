from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd
import requests

from battle_radar.config import AppConfig, DEFAULT_CONFIG
from battle_radar.database import SQLiteStore


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class MarketDataFetcher:
    def __init__(self, config: AppConfig | None = None, store: SQLiteStore | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store or SQLiteStore(self.config.db_path)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 BattleRadar/1.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )

    @staticmethod
    def is_main_board_code(code: str) -> bool:
        return code.startswith(("60", "000", "001", "002", "003"))

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        candidate_urls = [url]
        if "push2.eastmoney.com" in url:
            candidate_urls.extend(
                [
                    url.replace("push2.eastmoney.com", "push2delay.eastmoney.com"),
                    url.replace("push2.eastmoney.com", "82.push2.eastmoney.com"),
                    url.replace("push2.eastmoney.com", "82.push2delay.eastmoney.com"),
                ]
            )
        if "push2ex.eastmoney.com" in url:
            candidate_urls.append(url.replace("push2ex.eastmoney.com", "push2ex.eastmoney.com"))
        for candidate_url in dict.fromkeys(candidate_urls):
            for attempt in range(1, self.config.request_retries + 1):
                try:
                    response = self.session.get(candidate_url, params=params, timeout=self.config.request_timeout)
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning("请求失败，第 %s 次重试: %s params=%s error=%s", attempt, candidate_url, params, exc)
                    time.sleep(min(attempt, 3))
        raise RuntimeError(f"请求失败: {url} params={params} error={last_error}") from last_error

    def _fetch_paginated_clist(self, url: str, base_params: dict[str, Any]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        page = 1
        page_size = int(base_params.get("pz", 100))
        while True:
            params = dict(base_params)
            params["pn"] = str(page)
            try:
                payload = self._request_json(url, params)
            except RuntimeError:
                if frames:
                    logger.warning("分页抓取中断，保留已成功页: url=%s page=%s", url, page)
                    break
                raise
            data = payload.get("data") or {}
            records = data.get("diff") or []
            if not records:
                break
            frames.append(pd.DataFrame(records))
            total = data.get("total")
            if len(records) < page_size:
                break
            if total is not None and page * page_size >= int(total):
                break
            page += 1
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_stock_universe(self) -> pd.DataFrame:
        try:
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            common_params = {
                "pz": "5000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f12",
                "fields": "f12,f14,f13",
            }
            sh_df = self._fetch_paginated_clist(url, common_params | {"fs": "m:1+t:2,m:1+t:23"})
            sz_df = self._fetch_paginated_clist(url, common_params | {"fs": "m:0+t:6,m:0+t:80"})
            df = pd.concat([sh_df, sz_df], ignore_index=True)
            if not df.empty:
                df = df.rename(columns={"f12": "code", "f14": "name", "f13": "exchange_flag"})
                df["code"] = df["code"].astype(str).str.zfill(6)
                df = df[df["code"].map(self.is_main_board_code)].copy()
                df["exchange"] = df["code"].map(lambda code: "SH" if code.startswith("60") else "SZ")
                df["board"] = "main"
                return df[["code", "name", "exchange", "board"]].drop_duplicates().sort_values("code")
        except Exception as exc:  # noqa: BLE001
            logger.warning("东方财富股票池不可用，切换腾讯行情列表: %s", exc)

        try:
            tx_df = ak.stock_zh_a_spot_tx()
            if tx_df.empty:
                return pd.DataFrame(columns=["code", "name", "exchange", "board"])
            tx_df = tx_df.rename(columns={"name": "name"})
            tx_df["raw_code"] = tx_df["code"].astype(str)
            tx_df["code"] = tx_df["raw_code"].str.extract(r"(\d{6})", expand=False)
            tx_df = tx_df[tx_df["code"].notna()].copy()
            tx_df = tx_df[tx_df["code"].map(self.is_main_board_code)].copy()
            tx_df["exchange"] = tx_df["raw_code"].map(lambda x: "SH" if str(x).startswith("sh") else "SZ")
            tx_df["board"] = "main"
            return tx_df[["code", "name", "exchange", "board"]].drop_duplicates().sort_values("code")
        except Exception as exc:  # noqa: BLE001
            logger.error("腾讯行情股票池也不可用: %s", exc)
            return pd.DataFrame(columns=["code", "name", "exchange", "board"])

    def update_stock_universe(self) -> pd.DataFrame:
        universe = self.get_stock_universe()
        if not universe.empty:
            self.store.upsert_dataframe("stock_universe", universe, unique_columns=["code"])
        logger.info("股票池更新完成: %s", len(universe))
        return universe

    def fetch_daily_history(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.config.request_retries + 1):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                    timeout=self.config.request_timeout,
                )
                if df.empty:
                    return pd.DataFrame()
                renamed = df.rename(
                    columns={
                        "日期": "trade_date",
                        "股票代码": "code",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "换手率": "turnover",
                        "振幅": "amplitude",
                        "涨跌幅": "pct_chg",
                        "涨跌额": "chg",
                    }
                )
                renamed["code"] = renamed["code"].astype(str).str.zfill(6)
                return renamed[
                    [
                        "trade_date",
                        "code",
                        "open",
                        "close",
                        "high",
                        "low",
                        "volume",
                        "amount",
                        "turnover",
                        "amplitude",
                        "pct_chg",
                        "chg",
                    ]
                ]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("日线抓取失败，第 %s 次重试 symbol=%s error=%s", attempt, symbol, exc)
                time.sleep(min(attempt, 3))

        market_symbol = f"sh{symbol}" if symbol.startswith("60") else f"sz{symbol}"
        try:
            tx_df = ak.stock_zh_a_hist_tx(
                symbol=market_symbol,
                start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
                end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
                adjust="qfq",
            )
            if tx_df.empty:
                return pd.DataFrame()
            tx_df = tx_df.rename(
                columns={
                    "date": "trade_date",
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "volume": "volume",
                    "turnover": "turnover",
                    "amount": "amount",
                }
            )
            tx_df["code"] = symbol
            tx_df["amplitude"] = None
            tx_df["pct_chg"] = ((tx_df["close"] - tx_df["close"].shift(1)) / tx_df["close"].shift(1) * 100).round(2)
            tx_df["chg"] = (tx_df["close"] - tx_df["close"].shift(1)).round(2)
            tx_df["pct_chg"] = tx_df["pct_chg"].fillna(0)
            tx_df["chg"] = tx_df["chg"].fillna(0)
            return tx_df[
                [
                    "trade_date",
                    "code",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "turnover",
                    "amplitude",
                    "pct_chg",
                    "chg",
                ]
            ]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"抓取日线失败 symbol={symbol} error={exc}") from exc

    def update_daily_prices(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        max_symbols: int | None = None,
    ) -> int:
        universe = self.update_stock_universe()
        if max_symbols:
            universe = universe.head(max_symbols).copy()
        start_date = start_date or self.config.backfill_start_date
        end_date = end_date or date.today().strftime("%Y%m%d")
        inserted = 0
        for idx, row in universe.iterrows():
            history = self.fetch_daily_history(row["code"], start_date=start_date, end_date=end_date)
            if history.empty:
                continue
            history["name"] = row["name"]
            inserted += self.store.upsert_dataframe(
                "daily_prices",
                history,
                unique_columns=["trade_date", "code"],
            )
            if (idx + 1) % 100 == 0:
                logger.info("日线更新进度: %s/%s", idx + 1, len(universe))
        logger.info("日线更新完成: %s 条", inserted)
        return inserted

    def _normalize_limit_up(self, frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "code",
                    "name",
                    "latest_price",
                    "pct_chg",
                    "amount",
                    "circ_market_cap",
                    "total_market_cap",
                    "turnover",
                    "seal_amount",
                    "first_limit_time",
                    "last_limit_time",
                    "break_count",
                    "streak_stats",
                    "limit_up_count",
                    "industry",
                    "limit_up_reason",
                ]
            )
        normalized = frame.copy()
        normalized["trade_date"] = trade_date
        normalized["limit_up_reason"] = normalized.get("limit_up_reason")
        return normalized[
            [
                "trade_date",
                "code",
                "name",
                "latest_price",
                "pct_chg",
                "amount",
                "circ_market_cap",
                "total_market_cap",
                "turnover",
                "seal_amount",
                "first_limit_time",
                "last_limit_time",
                "break_count",
                "streak_stats",
                "limit_up_count",
                "industry",
                "limit_up_reason",
            ]
        ]

    def fetch_limit_up_pool(self, trade_date: str) -> pd.DataFrame:
        try:
            raw = ak.stock_zt_pool_em(date=trade_date)
            if not raw.empty:
                df = raw.rename(
                    columns={
                        "代码": "code",
                        "名称": "name",
                        "最新价": "latest_price",
                        "涨跌幅": "pct_chg",
                        "成交额": "amount",
                        "流通市值": "circ_market_cap",
                        "总市值": "total_market_cap",
                        "换手率": "turnover",
                        "封板资金": "seal_amount",
                        "首次封板时间": "first_limit_time",
                        "最后封板时间": "last_limit_time",
                        "炸板次数": "break_count",
                        "涨停统计": "streak_stats",
                        "连板数": "limit_up_count",
                        "所属行业": "industry",
                    }
                )
                df["code"] = df["code"].astype(str).str.zfill(6)
                df = df[df["code"].map(self.is_main_board_code)].copy()
                return self._normalize_limit_up(df, trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("akshare 涨停池接口失败，将切换兼容接口: %s", exc)

        url = "https://push2ex.eastmoney.com/getTopicZTPool"
        payload = self._request_json(
            url,
            {
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "dpt": "wz.ztzt",
                "Pageindex": "0",
                "pagesize": "10000",
                "sort": "fbt:asc",
                "date": trade_date,
            },
        )
        records = ((payload.get("data") or {}).get("pool")) or []
        rows: list[dict[str, Any]] = []
        for item in records:
            code = str(item["c"]).zfill(6)
            if not self.is_main_board_code(code):
                continue
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": item.get("n"),
                    "latest_price": (item.get("p") or 0) / 1000,
                    "pct_chg": item.get("zdp"),
                    "amount": item.get("amount"),
                    "circ_market_cap": item.get("ltsz"),
                    "total_market_cap": item.get("tshare"),
                    "turnover": item.get("hs"),
                    "seal_amount": item.get("fund"),
                    "first_limit_time": str(item.get("fbt", "")).zfill(6),
                    "last_limit_time": str(item.get("lbt", "")).zfill(6),
                    "break_count": item.get("zbc"),
                    "streak_stats": json.dumps(item.get("zttj", {}), ensure_ascii=False),
                    "limit_up_count": item.get("lbc"),
                    "industry": item.get("hybk"),
                    "limit_up_reason": None,
                }
            )
        return pd.DataFrame(rows)

    def fetch_limit_down_pool(self, trade_date: str) -> pd.DataFrame:
        url = "https://push2ex.eastmoney.com/getTopicDTPool"
        payload = self._request_json(
            url,
            {
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "dpt": "wz.ztzt",
                "Pageindex": "0",
                "pagesize": "10000",
                "sort": "fund:asc",
                "date": trade_date,
            },
        )
        records = ((payload.get("data") or {}).get("pool")) or []
        rows: list[dict[str, Any]] = []
        for item in records:
            code = str(item["c"]).zfill(6)
            if not self.is_main_board_code(code):
                continue
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": item.get("n"),
                    "latest_price": (item.get("p") or 0) / 1000,
                    "pct_chg": item.get("zdp"),
                    "amount": item.get("amount"),
                    "circ_market_cap": item.get("ltsz"),
                    "total_market_cap": item.get("tshare"),
                    "pe_dynamic": item.get("pe"),
                    "turnover": item.get("hs"),
                    "seal_amount": item.get("fund"),
                    "last_limit_time": str(item.get("lbt", "")).zfill(6),
                    "board_amount": item.get("fba"),
                    "consecutive_limit_down": item.get("lb"),
                    "break_count": item.get("zbc"),
                    "industry": item.get("hybk"),
                }
            )
        return pd.DataFrame(rows)

    def fetch_lhb(self, trade_date: str) -> pd.DataFrame:
        df = ak.stock_lhb_detail_em(start_date=trade_date, end_date=trade_date)
        if df.empty:
            return df
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        return df[df["代码"].map(self.is_main_board_code)].copy()

    def _fetch_board_list(self, board_type: str) -> pd.DataFrame:
        if board_type == "concept":
            fs = "m:90 t:3 f:!50"
        else:
            fs = "m:90 t:2 f:!50"
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pz": "500",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": fs,
            "fields": "f2,f3,f4,f8,f12,f14,f20,f104,f105,f128,f136",
        }
        df = self._fetch_paginated_clist(url, params)
        if df.empty:
            return pd.DataFrame()
        mapped = df.rename(
            columns={
                "f2": "latest_price",
                "f3": "pct_chg",
                "f4": "chg",
                "f8": "turnover",
                "f12": "board_code",
                "f14": "board_name",
                "f20": "total_market_cap",
                "f104": "up_count",
                "f105": "down_count",
                "f128": "leader_stock",
                "f136": "leader_pct_chg",
            }
        )
        mapped["board_type"] = board_type
        return mapped[
            [
                "board_type",
                "board_code",
                "board_name",
                "latest_price",
                "pct_chg",
                "chg",
                "total_market_cap",
                "turnover",
                "up_count",
                "down_count",
                "leader_stock",
                "leader_pct_chg",
            ]
        ]

    def fetch_board_snapshot(self, trade_date: str) -> pd.DataFrame:
        concept = self._fetch_board_list("concept")
        industry = self._fetch_board_list("industry")
        snapshot = pd.concat([concept, industry], ignore_index=True)
        if snapshot.empty:
            return snapshot
        snapshot["trade_date"] = trade_date
        snapshot["limit_up_count"] = 0.0
        return snapshot[
            [
                "trade_date",
                "board_type",
                "board_code",
                "board_name",
                "latest_price",
                "pct_chg",
                "chg",
                "total_market_cap",
                "turnover",
                "up_count",
                "down_count",
                "leader_stock",
                "leader_pct_chg",
                "limit_up_count",
            ]
        ]

    def fetch_board_members(self, board_type: str, board_code: str, board_name: str) -> pd.DataFrame:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pz": "500",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": f"b:{board_code} f:!50",
            "fields": "f12,f14",
        }
        df = self._fetch_paginated_clist(url, params)
        if df.empty:
            return pd.DataFrame(columns=["board_type", "board_code", "board_name", "code", "name"])
        df = df.rename(columns={"f12": "code", "f14": "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[df["code"].map(self.is_main_board_code)].copy()
        df["board_type"] = board_type
        df["board_code"] = board_code
        df["board_name"] = board_name
        return df[["board_type", "board_code", "board_name", "code", "name"]]

    def update_board_members(self, board_sample_size: int | None = None) -> int:
        sample_size = board_sample_size or self.config.board_sample_size
        boards = self.fetch_board_snapshot(date.today().strftime("%Y%m%d"))
        if boards.empty:
            return 0
        updated = 0
        boards = boards.sort_values("pct_chg", ascending=False).head(sample_size)
        for _, row in boards.iterrows():
            members = self.fetch_board_members(row["board_type"], row["board_code"], row["board_name"])
            if members.empty:
                continue
            updated += self.store.upsert_dataframe(
                "board_members",
                members,
                unique_columns=["board_type", "board_code", "code"],
            )
        logger.info("板块成分更新完成: %s", updated)
        return updated

    def update_market_snapshot(self, trade_date: str) -> dict[str, int]:
        limit_up = self.fetch_limit_up_pool(trade_date)
        limit_down = self.fetch_limit_down_pool(trade_date)
        try:
            board_snapshot = self.fetch_board_snapshot(trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.error("板块快照抓取失败 trade_date=%s error=%s", trade_date, exc)
            board_snapshot = pd.DataFrame(
                columns=[
                    "trade_date",
                    "board_type",
                    "board_code",
                    "board_name",
                    "latest_price",
                    "pct_chg",
                    "chg",
                    "total_market_cap",
                    "turnover",
                    "up_count",
                    "down_count",
                    "leader_stock",
                    "leader_pct_chg",
                    "limit_up_count",
                ]
            )
        counts = {
            "limit_up": self.store.upsert_dataframe(
                "limit_up_pool", limit_up, unique_columns=["trade_date", "code"]
            ),
            "limit_down": self.store.upsert_dataframe(
                "limit_down_pool", limit_down, unique_columns=["trade_date", "code"]
            ),
            "board_snapshot": self.store.upsert_dataframe(
                "board_snapshot",
                board_snapshot,
                unique_columns=["trade_date", "board_type", "board_code"],
            ),
        }
        return counts

    def recompute_board_limit_up_count(self, trade_date: str) -> int:
        sql = """
        WITH counts AS (
            SELECT bm.board_type, bm.board_code, COUNT(DISTINCT lup.code) AS limit_up_count
            FROM board_members bm
            LEFT JOIN limit_up_pool lup
              ON lup.code = bm.code
             AND lup.trade_date = ?
            GROUP BY bm.board_type, bm.board_code
        )
        UPDATE board_snapshot
           SET limit_up_count = COALESCE((
               SELECT counts.limit_up_count
               FROM counts
               WHERE counts.board_type = board_snapshot.board_type
                 AND counts.board_code = board_snapshot.board_code
           ), 0)
         WHERE trade_date = ?
        """
        conn = self.store.connect()
        try:
            cursor = conn.execute(sql, (trade_date, trade_date))
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def compute_market_metrics(self, trade_date: str) -> pd.DataFrame:
        price_stats = self.store.query(
            """
            SELECT
                SUM(CASE WHEN pct_chg > 0 THEN 1 ELSE 0 END) AS up_count,
                SUM(CASE WHEN pct_chg < 0 THEN 1 ELSE 0 END) AS down_count,
                SUM(CASE WHEN pct_chg = 0 THEN 1 ELSE 0 END) AS flat_count,
                SUM(amount) AS total_amount
            FROM daily_prices
            WHERE trade_date = ?
            """,
            (trade_date,),
        )
        limit_stats = self.store.query(
            """
            SELECT
                (SELECT COUNT(*) FROM limit_up_pool WHERE trade_date = ?) AS limit_up_count,
                (SELECT COUNT(*) FROM limit_down_pool WHERE trade_date = ?) AS limit_down_count,
                (SELECT COALESCE(MAX(limit_up_count), 0) FROM limit_up_pool WHERE trade_date = ?) AS board_height,
                (SELECT COUNT(*) FROM limit_up_pool WHERE trade_date = ? AND limit_up_count >= 2) AS streak_stock_count
            """,
            (trade_date, trade_date, trade_date, trade_date),
        )
        merged = pd.concat([price_stats, limit_stats], axis=1)
        merged["trade_date"] = trade_date
        merged["source"] = "daily_prices+limit_pools"
        merged = merged[
            [
                "trade_date",
                "up_count",
                "down_count",
                "flat_count",
                "limit_up_count",
                "limit_down_count",
                "total_amount",
                "board_height",
                "streak_stock_count",
                "source",
            ]
        ]
        self.store.upsert_dataframe("market_metrics", merged, unique_columns=["trade_date"])
        return merged

    def update_data_layer(
        self,
        trade_date: str | None = None,
        start_date: str | None = None,
        max_symbols: int | None = None,
        update_board_members: bool = True,
    ) -> dict[str, Any]:
        trade_date = trade_date or date.today().strftime("%Y%m%d")
        inserted_prices = self.update_daily_prices(start_date=start_date, end_date=trade_date, max_symbols=max_symbols)
        snapshot_counts = self.update_market_snapshot(trade_date)
        board_members = self.update_board_members() if update_board_members else 0
        self.recompute_board_limit_up_count(trade_date)
        metrics = self.compute_market_metrics(trade_date)
        return {
            "trade_date": trade_date,
            "price_rows": inserted_prices,
            "snapshot_counts": snapshot_counts,
            "board_member_rows": board_members,
            "market_metrics": metrics.to_dict(orient="records"),
            "config": asdict(self.config),
        }


if __name__ == "__main__":
    configure_logging()
    fetcher = MarketDataFetcher()
    result = fetcher.update_data_layer(
        trade_date=os.getenv("TRADE_DATE"),
        start_date=os.getenv("START_DATE"),
        max_symbols=int(os.getenv("MAX_SYMBOLS", "20")),
        update_board_members=os.getenv("UPDATE_BOARD_MEMBERS", "1") == "1",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
