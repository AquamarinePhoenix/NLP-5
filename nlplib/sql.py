def sql_drop_table(tablename: str) -> str:
    return f"""
        DROP TABLE IF EXISTS {tablename};
    """

def sql_select_table(tablename: str, *columns: str) -> str:
    cols = ", ".join(f'"{col}"' for col in columns)

    return f"""
        SELECT {cols}
        FROM {tablename}
    """

INSERT_METADATA: str = f"""
    INSERT INTO dimMetadata (RAW_INPUT, EMOTION_LABEL, TEXT_SAMPLE)
    VALUES (:raw_input, :emotion_label, :text_sample);
"""

CREATE_DIMENSION_METADATA: str = """
    CREATE TABLE IF NOT EXISTS dimMetadata (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        EMOTION_LABEL TEXT NULL,
        TEXT_SAMPLE TEXT NULL,
        RAW_INPUT TEXT NULL
    );
"""

CREATE_DIMENSION_METADATA_GROUPED: str = """
    CREATE TABLE IF NOT EXISTS dimMetadataGrouped (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        TEXT_SAMPLE TEXT UNIQUE,
        EMOTION_LABEL TEXT NULL
    );
"""

INSERT_METADATA_GROUPED: str = """
    INSERT INTO dimMetadataGrouped (TEXT_SAMPLE, EMOTION_LABEL)
    SELECT TEXT_SAMPLE, FIRST_VALUE(EMOTION_LABEL) OVER (PARTITION BY TEXT_SAMPLE ORDER BY TEXT_SAMPLE) AS EMOTION_LABEL
    FROM dimMetadata
    GROUP BY TEXT_SAMPLE;
"""

FIND_MULTIPLE_EMOTIONS = """
    SELECT TEXT_SAMPLE, EMOTIONS FROM (
        SELECT TEXT_SAMPLE, GROUP_CONCAT(EMOTION_LABEL, ';') AS EMOTIONS
        FROM dimMetadata
        GROUP BY TEXT_SAMPLE
    ) A
    WHERE EMOTIONS LIKE '%;%'
"""