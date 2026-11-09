#%%
import subprocess
import sys

from nlplib.log import clear_log, clear_output_pngs, write_log
from nlplib.data import prepare_data
from nlplib.db import (
    get_connection,
    clear_database,
    reset_table,
    find_multiple_emotions,
    write_metadata,
    get_transformed_text_by_emotion,
    get_text_by_emotion,
    get_text_length_distribution,
    save_representations,
    get_dataset_statistics,
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
from nlplib.plot import (
    plot_emotion_word_clouds,
    plot_line_chart,
    plot_pos_distribution,
    plot_umap_representations,
)
from nlplib.transform import get_pos_distribution, perform_preprocess
from nlplib.features import extract_representations, stratified_text_split
from nlplib.classification import run_emotion_classification
from nlplib.config import EMOTIONS
from nlplib.statistics import log_text_length_statistics
import pandas as pd
from scipy.sparse import issparse
def main():
    #%%
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )

    #1. Text dataset preparation
    clear_log()
    clear_output_pngs()
    conn, cursor = get_connection()
    clear_database(conn, cursor)

    data = prepare_data()
    write_log(data)

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
    dataset_statistics = get_dataset_statistics(conn, cursor)
    write_log(
        "Dataset statistics: "
        f"raw_rows={dataset_statistics['raw_rows']}, "
        f"unique_texts_after_grouping={dataset_statistics['unique_rows']}, "
        f"duplicates_removed={dataset_statistics['duplicates_removed']}, "
        f"multi_label_texts_excluded={dataset_statistics['multi_label_texts']}, "
        f"class_distribution={dataset_statistics['class_distribution']}"
    )
    write_log(
        "Multi-label handling: texts with multiple emotion labels are excluded "
        "from analysis by dimMetadataGrouped; one label is retained per unique "
        "text using FIRST_VALUE."
    )
    write_metadata(conn,cursor)
#%%
    #2. Text Analysis
    reset_table(
        conn,
        cursor,
        "dimTextLengthDistribution",
        CREATE_DIMENSION_TEXT_LENGTH_DISTRIBUTION,
        INSERT_TEXT_LENGTH_DISTRIBUTION,
    )

    distribution = get_text_length_distribution(conn, cursor)
    log_text_length_statistics(distribution)
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
    #3. Text Representation (Feature Extraction)
    feature_rows = [
        (emotion, text)
        for emotion, _, text in get_text_by_emotion(conn, cursor)
    ]
    feature_split = stratified_text_split(feature_rows)
    representations = extract_representations(feature_split)
    save_representations(conn, cursor, feature_split, representations)
    plot_umap_representations(representations, feature_split)
    write_log(
        "Feature extraction shapes: "
        f"TF-IDF train={representations['tfidf_train'].shape}, "
        f"test={representations['tfidf_test'].shape}; "
        f"static train={representations['static_train'].shape}, "
        f"test={representations['static_test'].shape}; "
        f"contextual train={representations['contextual_train'].shape}, "
        f"test={representations['contextual_test'].shape}"
    )
    for representation_name, train_features, test_features in (
        ("frequency", representations["tfidf_train"], representations["tfidf_test"]),
        ("static", representations["static_train"], representations["static_test"]),
        ("contextual", representations["contextual_train"], representations["contextual_test"]),
    ):
        train_size = train_features.shape[0] * train_features.shape[1]
        test_size = test_features.shape[0] * test_features.shape[1]
        train_nonzero = (
            train_features.nnz
            if issparse(train_features)
            else (train_features != 0).sum()
        )
        test_nonzero = (
            test_features.nnz
            if issparse(test_features)
            else (test_features != 0).sum()
        )
        write_log(
            f"{representation_name} representation report statistics: "
            f"dimensions={train_features.shape[1]}, "
            f"train_sparsity={1 - train_nonzero / train_size:.4f}, "
            f"test_sparsity={1 - test_nonzero / test_size:.4f}; "
            f"semantic_information={'surface term weighting' if representation_name == 'frequency' else 'distributional context' if representation_name == 'static' else 'contextual sentence semantics'}"
        )
#%%
    #4. Emotion Classification
    run_emotion_classification(
        feature_split,
        representations,
    )
    
#%%
if __name__ == "__main__":
    main()
    write_log("The main process ended successfully.")