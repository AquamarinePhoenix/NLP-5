import sqlite3
import csv
from nlplib.config import DB_FILE, OUTPUT_FILE
from nlplib.sql import (
    CREATE_DIMENSION_METADATA,
    INSERT_METADATA,
    FIND_MULTIPLE_EMOTIONS,
    CREATE_DIMENSION_METADATA_GROUPED,
    INSERT_METADATA_GROUPED,
    sql_select_table,
    sql_drop_table
)

def get_connection() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    return conn, cursor

def drop_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor, tablename: str) -> None:
    cursor.execute(sql_drop_table(tablename))
    conn.commit()

def create_dimMetadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    cursor.execute(CREATE_DIMENSION_METADATA)
    conn.commit()

def insert_metadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor, rows: list[dict[str, str]],) -> None:
    for row in rows:
        cursor.execute(INSERT_METADATA, row)
    conn.commit()

def find_multiple_emotions(conn: sqlite3.Connection, cursor: sqlite3.Cursor,) -> list[dict[str, str]]:
    cursor.execute(FIND_MULTIPLE_EMOTIONS)
    rows = cursor.fetchall()
    conn.commit()

    return rows

def create_dimMetadataGrouped(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    cursor.execute(CREATE_DIMENSION_METADATA_GROUPED)
    conn.commit()

def insert_metadata_grouped(conn: sqlite3.Connection, cursor: sqlite3.Cursor,) -> None:
    cursor.execute(INSERT_METADATA_GROUPED)
    conn.commit()

def write_metadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor,) -> None:
    cursor.execute(sql_select_table("dimMetadataGrouped", "TEXT_SAMPLE", "EMOTION_LABEL"))
    
    headers = [description[0] for description in cursor.description]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(cursor.fetchall())