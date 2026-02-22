import sqlite3
import os
import json

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "app.db")


def connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            draw_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_datetime_utc   TEXT NOT NULL UNIQUE,
            draw_datetime_local TEXT NOT NULL,
            game           TEXT NOT NULL DEFAULT 'Powerball',
            numbers_json   TEXT NOT NULL,
            powerball      INTEGER NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS planet_positions (
            draw_id       INTEGER PRIMARY KEY,
            location_name TEXT,
            latitude      REAL,
            longitude     REAL,
            altitude_m    REAL DEFAULT 0.0,
            positions_json TEXT NOT NULL,
            FOREIGN KEY (draw_id) REFERENCES draws(draw_id) ON DELETE CASCADE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS planet_features (
            draw_id       INTEGER PRIMARY KEY,
            features_json TEXT NOT NULL,
            FOREIGN KEY (draw_id) REFERENCES draws(draw_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


def get_draw_count() -> int:
    conn = connect()
    count = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    conn.close()
    return count


def get_position_count() -> int:
    conn = connect()
    count = conn.execute("SELECT COUNT(*) FROM planet_positions").fetchone()[0]
    conn.close()
    return count


def get_feature_count() -> int:
    conn = connect()
    count = conn.execute("SELECT COUNT(*) FROM planet_features").fetchone()[0]
    conn.close()
    return count


def get_draws_without_positions():
    conn = connect()
    rows = conn.execute("""
        SELECT d.draw_id, d.draw_datetime_utc
        FROM draws d
        LEFT JOIN planet_positions pp ON d.draw_id = pp.draw_id
        WHERE pp.draw_id IS NULL
        ORDER BY d.draw_datetime_utc
    """).fetchall()
    conn.close()
    return rows


def get_draws_without_features():
    conn = connect()
    rows = conn.execute("""
        SELECT pp.draw_id, pp.positions_json
        FROM planet_positions pp
        LEFT JOIN planet_features pf ON pp.draw_id = pf.draw_id
        WHERE pf.draw_id IS NULL
        ORDER BY pp.draw_id
    """).fetchall()
    conn.close()
    return rows


def load_draws_df():
    import pandas as pd
    conn = connect()
    df = pd.read_sql_query("SELECT * FROM draws ORDER BY draw_datetime_utc", conn)
    conn.close()
    return df


def load_features_df():
    import pandas as pd
    conn = connect()
    df = pd.read_sql_query("SELECT * FROM planet_features ORDER BY draw_id", conn)
    conn.close()
    return df


def load_positions_df():
    import pandas as pd
    conn = connect()
    df = pd.read_sql_query("SELECT * FROM planet_positions ORDER BY draw_id", conn)
    conn.close()
    return df


def clear_all_data():
    conn = connect()
    conn.execute("DELETE FROM planet_features")
    conn.execute("DELETE FROM planet_positions")
    conn.execute("DELETE FROM draws")
    conn.commit()
    conn.close()
