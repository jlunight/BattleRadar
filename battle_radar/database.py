from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS stock_universe (
        code TEXT PRIMARY KEY,
        name TEXT,
        exchange TEXT,
        board TEXT,
        list_date TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        trade_date TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        open REAL,
        close REAL,
        high REAL,
        low REAL,
        volume REAL,
        amount REAL,
        turnover REAL,
        amplitude REAL,
        pct_chg REAL,
        chg REAL,
        market_cap REAL,
        circ_market_cap REAL,
        PRIMARY KEY (trade_date, code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS limit_up_pool (
        trade_date TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        latest_price REAL,
        pct_chg REAL,
        amount REAL,
        circ_market_cap REAL,
        total_market_cap REAL,
        turnover REAL,
        seal_amount REAL,
        first_limit_time TEXT,
        last_limit_time TEXT,
        break_count REAL,
        streak_stats TEXT,
        limit_up_count REAL,
        industry TEXT,
        limit_up_reason TEXT,
        PRIMARY KEY (trade_date, code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS limit_down_pool (
        trade_date TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        latest_price REAL,
        pct_chg REAL,
        amount REAL,
        circ_market_cap REAL,
        total_market_cap REAL,
        pe_dynamic REAL,
        turnover REAL,
        seal_amount REAL,
        last_limit_time TEXT,
        board_amount REAL,
        consecutive_limit_down REAL,
        break_count REAL,
        industry TEXT,
        PRIMARY KEY (trade_date, code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS board_snapshot (
        trade_date TEXT NOT NULL,
        board_type TEXT NOT NULL,
        board_code TEXT NOT NULL,
        board_name TEXT NOT NULL,
        latest_price REAL,
        pct_chg REAL,
        chg REAL,
        total_market_cap REAL,
        turnover REAL,
        up_count REAL,
        down_count REAL,
        leader_stock TEXT,
        leader_pct_chg REAL,
        limit_up_count REAL,
        PRIMARY KEY (trade_date, board_type, board_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS board_members (
        board_type TEXT NOT NULL,
        board_code TEXT NOT NULL,
        board_name TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (board_type, board_code, code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_metrics (
        trade_date TEXT PRIMARY KEY,
        up_count REAL,
        down_count REAL,
        flat_count REAL,
        limit_up_count REAL,
        limit_down_count REAL,
        total_amount REAL,
        board_height REAL,
        streak_stock_count REAL,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS emotion_scores (
        trade_date TEXT PRIMARY KEY,
        emotion_score REAL,
        cycle_label TEXT,
        detail_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_signals (
        trade_date TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        score REAL,
        tier TEXT,
        buy_mode TEXT,
        buy_price REAL,
        stop_loss REAL,
        max_position REAL,
        reason TEXT,
        cycle_label TEXT,
        metrics_json TEXT,
        PRIMARY KEY (trade_date, strategy_name, code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        signal_date TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT,
        buy_mode TEXT,
        buy_price REAL,
        next_trade_date TEXT,
        next_open REAL,
        next_high REAL,
        max_return_pct REAL,
        success INTEGER,
        pnl_ratio REAL,
        sample_tag TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (signal_date, strategy_name, code)
    )
    """,
]


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        conn = self.connect()
        try:
            for statement in DDL_STATEMENTS:
                conn.execute(statement)
            conn.commit()
        finally:
            conn.close()

    def upsert_dataframe(
        self,
        table_name: str,
        frame: pd.DataFrame,
        unique_columns: Iterable[str],
    ) -> int:
        if frame.empty:
            return 0
        rows = frame.copy()
        columns = list(rows.columns)
        placeholders = ", ".join(["?"] * len(columns))
        column_sql = ", ".join(columns)
        update_columns = [column for column in columns if column not in set(unique_columns)]
        update_sql = ", ".join([f"{column}=excluded.{column}" for column in update_columns])
        sql = (
            f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(unique_columns)}) DO UPDATE SET {update_sql}"
        )
        conn = self.connect()
        try:
            conn.executemany(sql, rows.where(pd.notna(rows), None).values.tolist())
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def query(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        conn = self.connect()
        try:
            return pd.read_sql_query(sql, conn, params=params or ())
        finally:
            conn.close()
