DROP_METADATA: str = """
    DROP TABLE IF EXISTS dimMetadata;
"""

CREATE_DIMENSION_METADATA: str = """
    CREATE TABLE IF NOT EXISTS dimMetadata (
        ID__ INTEGER PRIMARY KEY AUTOINCREMENT,
        RAW_INPUT TEXT NULL,
        EMOTION_LABEL TEXT NULL,
        TEXT_SAMPLE TEXT NULL
    );
"""

INSERT_METADATA: str = """
    INSERT INTO dimMetadata (RAW_INPUT, EMOTION_LABEL, TEXT_SAMPLE)
    VALUES (:raw_input, :emotion_label, :text_sample);
"""