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
        EMOTION_LABEL TEXT NULL,
        TRANSFORMED_TEXT TEXT NULL
    );
"""

INSERT_METADATA_GROUPED: str = """
    INSERT INTO dimMetadataGrouped (TEXT_SAMPLE, EMOTION_LABEL, TRANSFORMED_TEXT)
    SELECT TEXT_SAMPLE, FIRST_VALUE(EMOTION_LABEL) OVER (PARTITION BY TEXT_SAMPLE ORDER BY TEXT_SAMPLE), NULL
    FROM dimMetadata
    GROUP BY TEXT_SAMPLE;
"""

UPDATE_METADATA_GROUPED_TRANSFORMED: str = """
    UPDATE dimMetadataGrouped
    SET TRANSFORMED_TEXT = :transformed_text
    WHERE TEXT_SAMPLE = :text_sample;
"""

FIND_MULTIPLE_EMOTIONS = """
    SELECT TEXT_SAMPLE, EMOTIONS FROM (
        SELECT TEXT_SAMPLE, GROUP_CONCAT(EMOTION_LABEL, ';') AS EMOTIONS
        FROM dimMetadata
        GROUP BY TEXT_SAMPLE
    ) A
    WHERE EMOTIONS LIKE '%;%'
"""

CREATE_DIMENSION_TEXT_LENGTH_DISTRIBUTION: str = """
    CREATE TABLE IF NOT EXISTS dimTextLengthDistribution (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        EMOTION_LABEL TEXT NULL,
        TEXT_LENGTH INTEGER NULL,
        RECORD_COUNT INTEGER NULL
    );
"""

INSERT_TEXT_LENGTH_DISTRIBUTION: str = """
    INSERT INTO dimTextLengthDistribution (
        EMOTION_LABEL,
        TEXT_LENGTH,
        RECORD_COUNT
    )
    SELECT
        EMOTION_LABEL,
        LENGTH(TEXT_SAMPLE) AS TEXT_LENGTH,
        COUNT(*) AS RECORD_COUNT
    FROM dimMetadataGrouped
    GROUP BY EMOTION_LABEL, TEXT_LENGTH
"""

CREATE_DIMENSION_EMBEDDINGS: str = """
    CREATE TABLE IF NOT EXISTS dimEmbeddings (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        TEXT_SAMPLE TEXT NULL,
        EMBEDDING TEXT NULL,
        MODEL_TYPE TEXT NULL,
        IS_SPARSE INTEGER NOT NULL
    );
"""

INSERT_EMBEDDINGS: str = """
    INSERT INTO dimEmbeddings (
        TEXT_SAMPLE,
        EMBEDDING,
        MODEL_TYPE,
        IS_SPARSE
    )
    VALUES (
        :text_sample,
        :embedding,
        :model_type,
        :is_sparse
    );
"""

FIND_TEXT_LENGTH_DISTRIBUTION = """
    SELECT
        B.EMOTION_DEFINITION,
        A.TEXT_LENGTH,
        A.RECORD_COUNT
    FROM dimTextLengthDistribution AS A
    INNER JOIN dimEmotions AS B ON A.EMOTION_LABEL = B.EMOTION_LABEL
    ORDER BY A.EMOTION_LABEL, A.TEXT_LENGTH;
"""

FIND_TRANSFORMED_TEXT_BY_EMOTION = """
    SELECT EMOTION_LABEL, TRANSFORMED_TEXT
    FROM dimMetadataGrouped
    WHERE TRANSFORMED_TEXT IS NOT NULL
      AND TRANSFORMED_TEXT <> ''
    ORDER BY EMOTION_LABEL;
"""

FIND_TEXT_BY_EMOTION = """
    SELECT
        A.EMOTION_LABEL,
        B.EMOTION_DEFINITION,
        A.TEXT_SAMPLE
    FROM dimMetadataGrouped AS A
    INNER JOIN dimEmotions AS B ON A.EMOTION_LABEL = B.EMOTION_LABEL
    WHERE A.TEXT_SAMPLE IS NOT NULL
      AND A.TEXT_SAMPLE <> ''
    ORDER BY A.EMOTION_LABEL;
"""

CREATE_DIMENSION_EMOTIONS: str = """
    CREATE TABLE IF NOT EXISTS dimEmotions (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        EMOTION_LABEL TEXT NULL,
        EMOTION_DEFINITION TEXT NULL
    );
"""

INSERT_EMOTIONS: str = f"""
    INSERT INTO dimEmotions (EMOTION_LABEL, EMOTION_DEFINITION)
    VALUES (:emotion_label, :emotion_definition);
"""