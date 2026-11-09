import sqlite3
import csv
import json
import numpy as np
from collections.abc import Iterable
from scipy.sparse import issparse
from nlplib.config import DB_FILE, OUTPUT_FILE
from nlplib.sql import (
    FIND_MULTIPLE_EMOTIONS,
    FIND_TRANSFORMED_TEXT_BY_EMOTION,
    FIND_TEXT_BY_EMOTION,
    FIND_TEXT_LENGTH_DISTRIBUTION,
    UPDATE_METADATA_GROUPED_TRANSFORMED,
    sql_select_table,
    sql_drop_table,
    CREATE_DIMENSION_EMBEDDINGS,
    INSERT_EMBEDDINGS,
)

def get_connection() -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    return conn, cursor


def clear_database(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> None:
    table_names = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    for (table_name,) in table_names:
        cursor.execute(f'DELETE FROM "{table_name}"')
    conn.commit()


def reset_table(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    tablename: str,
    create_query: str,
    insert_query: str,
    rows: Iterable[dict[str, object]] | None = None,
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


def get_dataset_statistics(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
) -> dict[str, object]:
    raw_rows = cursor.execute(
        "SELECT COUNT(*) FROM dimMetadata"
    ).fetchone()[0]
    unique_rows = cursor.execute(
        "SELECT COUNT(*) FROM dimMetadataGrouped"
    ).fetchone()[0]
    multi_label_rows = cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT TEXT_SAMPLE
            FROM dimMetadata
            GROUP BY TEXT_SAMPLE
            HAVING COUNT(DISTINCT EMOTION_LABEL) > 1
        )
        """
    ).fetchone()[0]
    class_distribution = dict(cursor.execute(
        """
        SELECT EMOTION_LABEL, COUNT(*)
        FROM dimMetadataGrouped
        GROUP BY EMOTION_LABEL
        ORDER BY EMOTION_LABEL
        """
    ).fetchall())
    return {
        "raw_rows": raw_rows,
        "unique_rows": unique_rows,
        "duplicates_removed": raw_rows - unique_rows,
        "multi_label_texts": multi_label_rows,
        "class_distribution": class_distribution,
    }


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


def save_representations(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    split: dict[str, list[str]],
    representations: dict,
) -> None:
    model_embeddings = (
        (
            "frequency",
            representations["tfidf_train"],
            representations["tfidf_test"],
        ),
        (
            "static",
            representations["static_train"],
            representations["static_test"],
        ),
        (
            "contextual",
            representations["contextual_train"],
            representations["contextual_test"],
        ),
    )
    texts = split["train_texts"] + split["test_texts"]

    cursor.execute(sql_drop_table("dimEmbeddings"))
    cursor.execute(CREATE_DIMENSION_EMBEDDINGS)
    for model_type, train_embeddings, test_embeddings in model_embeddings:
        cursor.executemany(
            INSERT_EMBEDDINGS,
            _embedding_rows(
                model_type,
                texts,
                train_embeddings,
                test_embeddings,
            ),
        )
    conn.commit()


def _embedding_rows(model_type, texts, train_embeddings, test_embeddings):
    for text, embedding in zip(
        texts,
        _iter_embedding_rows(train_embeddings, test_embeddings),
    ):
        if issparse(embedding):
            serialized_embedding = {
                "size": embedding.shape[1],
                "indices": embedding.indices.tolist(),
                "values": embedding.data.tolist(),
            }
            is_sparse = True
        else:
            serialized_embedding = embedding.tolist()
            is_sparse = np.count_nonzero(embedding) < embedding.size / 2
        yield {
            "text_sample": text,
            "embedding": json.dumps(serialized_embedding),
            "model_type": model_type,
            "is_sparse": int(is_sparse),
        }


def _iter_embedding_rows(train_embeddings, test_embeddings):
    for matrix in (train_embeddings, test_embeddings):
        for row in matrix:
            if issparse(row):
                yield row.tocsr()
            else:
                yield np.asarray(row).ravel()


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
