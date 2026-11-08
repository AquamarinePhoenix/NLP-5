#%%
from nlplib.log import clear_log, write_log
from nlplib.data import read_raw_data, EMOTIONS
from nlplib.db import (
    get_connection,
    reset_table,
    find_multiple_emotions,
    write_metadata,
    get_transformed_text_by_emotion,
    get_text_by_emotion,
    get_text_length_distribution,
)
from nlplib.sql import (
    CREATE_DIMENSION_METADATA,
    INSERT_METADATA,
    CREATE_DIMENSION_METADATA_GROUPED,
    INSERT_METADATA_GROUPED,
    CREATE_DIMENSION_EMOTIONS,
    INSERT_EMOTIONS,
    CREATE_DIMENSION_TEXT_LENGTH_DISTRIBUTION,
    INSERT_TEXT_LENGTH_DISTRIBUTION,
    sql_select_table
)
from nlplib.plot import plot_emotion_word_clouds, plot_line_chart, plot_pos_distribution
from nlplib.transform import get_pos_distribution, perform_preprocess
import pandas as pd
#%%
def main():
    #%%
    clear_log()
    data = read_raw_data()
    write_log(data)

    conn, cursor = get_connection()

    reset_table(
        conn,
        cursor,
        "dimEmotions",
        CREATE_DIMENSION_EMOTIONS,
        INSERT_EMOTIONS,
        EMOTIONS
    )

    reset_table(
        conn,
        cursor,
        "dimMetadata",
        CREATE_DIMENSION_METADATA,
        INSERT_METADATA,
        data,
    )
#%%
    rows = find_multiple_emotions(conn, cursor,)
    write_log(rows)

    reset_table(
        conn,
        cursor,
        "dimMetadataGrouped",
        CREATE_DIMENSION_METADATA_GROUPED,
        INSERT_METADATA_GROUPED,
    )
    write_metadata(conn,cursor)

    reset_table(
        conn,
        cursor,
        "dimTextLengthDistribution",
        CREATE_DIMENSION_TEXT_LENGTH_DISTRIBUTION,
        INSERT_TEXT_LENGTH_DISTRIBUTION,
    )

    distribution = get_text_length_distribution(conn, cursor)
    plot_line_chart(
        "Text length distribution by emotion",
        "Text length",
        "Count",
        "Emotion class",
        distribution
    )

#%%
    query = sql_select_table("dimMetadataGrouped", "TEXT_SAMPLE")
    text_df = pd.read_sql_query(query, conn)

    text_list = text_df["TEXT_SAMPLE"].dropna().tolist()
    processed_rows = perform_preprocess(conn, cursor, text_list,)

    write_log(f"Processed texts: {processed_rows}")
#%%
    transformed_rows = get_transformed_text_by_emotion(conn, cursor)
    frequent_words = plot_emotion_word_clouds(transformed_rows)
    
    for emotion, words in frequent_words.items():
        write_log(f"Most frequent words for {emotion}: {words}")

#%%
    pos_rows = get_pos_distribution(get_text_by_emotion(conn, cursor))
    plot_pos_distribution(pos_rows)
    write_log("POS distributions by emotion:")
    
    for emotion in sorted({row[0] for row in pos_rows}):
        emotion_rows = [row for row in pos_rows if row[0] == emotion]
        write_log(
            f"{emotion}: "
            + ", ".join(f"{pos}={percentage:.2f}%" for _, pos, _, percentage in emotion_rows)
        )

#%%
if __name__ == "__main__":
    main()
    write_log("The main process ended successfully.")