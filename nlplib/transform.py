import spacy
from spacy.cli import download as download_spacy_model
from collections.abc import Iterable
from multiprocessing import current_process
from nlplib.log import write_log
from nlplib.db import update_transformed_metadata
from concurrent.futures import ProcessPoolExecutor

_worker_nlp = None
_worker_stop_words = None
_worker_id = None

def load_spacy_model():
    try:
        nlp = spacy.load("lt_core_news_sm")
    except OSError:
        download_spacy_model("lt_core_news_sm")
        nlp = spacy.load("lt_core_news_sm")

    return nlp

def _lemmatize_doc(doc, stop_words) -> list[str]:
    words = []

    for token in doc:

        if not token.is_alpha:
            continue

        lemma = token.lemma_.lower()

        if (
            lemma not in stop_words
            and len(lemma) > 2
        ):
            words.append(lemma)

    return words


def lemmatization_stopwords(nlp, text: str, stop_words) -> list[str]:
    if not isinstance(text, str):
        return []

    return _lemmatize_doc(nlp(text.lower()), stop_words)


def lemmatization_stopwords_batch(
    nlp,
    texts: Iterable[str],
    stop_words,
    batch_size: int = 64,
) -> list[list[str]]:
    valid_texts = (text.lower() for text in texts if isinstance(text, str))
    return [
        _lemmatize_doc(doc, stop_words)
        for doc in nlp.pipe(valid_texts, batch_size=batch_size)
    ]


def initialize_text_worker(worker_count: int = 4) -> None:
    global _worker_nlp, _worker_stop_words, _worker_id
    _worker_nlp = load_spacy_model()
    _worker_stop_words = _worker_nlp.Defaults.stop_words
    process_identity = current_process()._identity[0] if current_process()._identity else 1
    _worker_id = (process_identity - 1) % worker_count + 1


def process_text_batch(
    batch: tuple[int, list[str]],
) -> tuple[int, int, list[list[str]]]:
    batch_number, texts = batch
    if (
        _worker_nlp is None
        or _worker_stop_words is None
        or _worker_id is None
    ):
        raise RuntimeError(
            f"Text worker {current_process().name} was not initialized"
        )

    return (
        batch_number,
        _worker_id,
        lemmatization_stopwords_batch(
            _worker_nlp,
            texts,
            _worker_stop_words,
        ),
    )

def process_pos_batch(
    batch: tuple[int, list[tuple[str, str, str]]],
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    batch_number, rows = batch
    if _worker_nlp is None or _worker_id is None:
        raise RuntimeError(
            f"POS worker {current_process().name} was not initialized"
        )

    valid_rows = [
        (emotion_label, emotion_definition, text)
        for emotion_label, emotion_definition, text in rows
        if isinstance(text, str) and text
    ]
    counts: dict[str, dict[str, int]] = {}
    for (emotion, _), doc in zip(
        [
            (f"{label} - {definition}", text)
            for label, definition, text in valid_rows
        ],
        _worker_nlp.pipe(
            (text for _, _, text in valid_rows),
            batch_size=64,
        ),
    ):
        emotion_counts = counts.setdefault(emotion, {})
        for token in doc:
            if token.is_alpha and token.pos_:
                emotion_counts[token.pos_] = emotion_counts.get(token.pos_, 0) + 1

    return batch_number, _worker_id, len(valid_rows), counts


def get_pos_distribution(
    rows: Iterable[tuple[str, str, str]],
    batch_size: int = 250,
    worker_count: int = 4,
) -> list[tuple[str, str, int, float]]:
    counts: dict[str, dict[str, int]] = {}

    rows_list = list(rows)
    batches = [
        (batch_number, rows_list[start:start + batch_size])
        for batch_number, start in enumerate(
            range(0, len(rows_list), batch_size),
            1,
        )
    ]

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=initialize_text_worker,
        initargs=(worker_count,),
    ) as executor:
        processed_rows = 0
        for batch_number, worker_id, batch_rows, batch_counts in executor.map(
            process_pos_batch,
            batches,
        ):
            for emotion, pos_counts in batch_counts.items():
                emotion_counts = counts.setdefault(emotion, {})
                for pos, count in pos_counts.items():
                    emotion_counts[pos] = emotion_counts.get(pos, 0) + count
            processed_rows += batch_rows
            write_log(
                f"POS worker {worker_id} processed batch "
                f"{batch_number}/{len(batches)} "
                f"({processed_rows}/{len(rows_list)} rows)"
            )

    distribution: list[tuple[str, str, int, float]] = []
    for emotion in sorted(counts):
        total = sum(counts[emotion].values())
        if total == 0:
            continue
        for pos, count in sorted(counts[emotion].items()):
            distribution.append((emotion, pos, count, count / total * 100))

    return distribution


def perform_preprocess(
    conn,
    cursor,
    text_list,
    batch_size: int = 1000,
    worker_count: int = 4,
):

    batches = [
        (batch_number, text_list[start:start + batch_size])
        for batch_number, start in enumerate(range(0, len(text_list), batch_size), 1)
    ]
    processed_rows = 0

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=initialize_text_worker,
        initargs=(worker_count,),
    ) as executor:
        for batch_number, worker_id, transformed_batch in executor.map(
            process_text_batch,
            batches,
        ):
            updates = [
                {
                    "text_sample": text,
                    "transformed_text": " ".join(transformed),
                }
                for text, transformed in zip(
                    batches[batch_number - 1][1],
                    transformed_batch,
                )
            ]
            update_transformed_metadata(conn, cursor, updates)
            processed_rows += len(updates)

            write_log(
                f"Worker {worker_id} processed batch {batch_number}/{len(batches)} "
                f"({processed_rows}/{len(text_list)} rows)"
            )

    return processed_rows