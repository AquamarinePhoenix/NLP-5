import sqlite3
from nlplib.config import DB_FILE
from nlplib.sql import CREATE_DIMENSION_METADATA, DROP_METADATA, INSERT_METADATA

def get_connection() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    return conn, cursor

def drop_dimMetadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    cursor.execute(DROP_METADATA)
    conn.commit()

def create_dimMetadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    cursor.execute(CREATE_DIMENSION_METADATA)
    conn.commit()

def insert_metadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor, rows: list[dict[str, str]],) -> None:
    for row in rows:
        cursor.execute(INSERT_METADATA, row)

    conn.commit()