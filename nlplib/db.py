import sqlite3
import csv
from nlplib.config import DB_FILE, OUTPUT_FILE
from nlplib.sql import (
    FIND_MULTIPLE_EMOTIONS,
    FIND_TRANSFORMED_TEXT_BY_EMOTION,
    FIND_TEXT_BY_EMOTION,
    FIND_TEXT_LENGTH_DISTRIBUTION,
    UPDATE_METADATA_GROUPED_TRANSFORMED,
    sql_select_table,
    sql_drop_table
)

def get_connection() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    return conn, cursor

def reset_table(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    tablename: str,
    create_query: str,
    insert_query: str,
    rows: list[dict[str, str]] | None = None,
) -> None:
    cursor.execute(sql_drop_table(tablename))
    cursor.execute(create_query)

    if rows is None:
        cursor.execute(insert_query)
    else:
        cursor.executemany(insert_query, rows)

    conn.commit()

def find_multiple_emotions(conn: sqlite3.Connection, cursor: sqlite3.Cursor,) -> list[dict[str, str]]:
    cursor.execute(FIND_MULTIPLE_EMOTIONS)
    rows = cursor.fetchall()
    conn.commit()

    return rows

def write_metadata(conn: sqlite3.Connection, cursor: sqlite3.Cursor,) -> None:
    cursor.execute(sql_select_table(
        "dimMetadataGrouped",
        "TEXT_SAMPLE",
        "EMOTION_LABEL",
        "TRANSFORMED_TEXT",
    ))
    
    headers = [description[0] for description in cursor.description]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(cursor.fetchall())


def update_transformed_metadata(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    rows: list[dict[str, str]],
) -> None:
    cursor.executemany(UPDATE_METADATA_GROUPED_TRANSFORMED, rows)
    conn.commit()

def get_text_length_distribution(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> list[tuple[str, int, int]]:
    cursor.execute(FIND_TEXT_LENGTH_DISTRIBUTION)
    rows = cursor.fetchall()
    conn.commit()
    return rows


def get_transformed_text_by_emotion(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> list[tuple[str, str]]:
    cursor.execute(FIND_TRANSFORMED_TEXT_BY_EMOTION)
    rows = cursor.fetchall()
    conn.commit()
    return rows


def get_text_by_emotion(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> list[tuple[str, str, str]]:
    cursor.execute(FIND_TEXT_BY_EMOTION)
    rows = cursor.fetchall()
    conn.commit()
    return rows
