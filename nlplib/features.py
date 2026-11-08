from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


TextSplit = dict[str, list[str]]
FeatureRepresentations = dict[str, Any]


def stratified_text_split(
    rows: list[tuple[str, str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TextSplit:
    texts = [text for _, text in rows]
    labels = [label for label, _ in rows]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return {
        "train_texts": train_texts,
        "test_texts": test_texts,
        "train_labels": train_labels,
        "test_labels": test_labels,
    }


def fit_static_embeddings(
    texts: list[str],
    dimensions: int = 100,
    window_size: int = 2,
) -> tuple[dict[str, int], np.ndarray]:
    tokens_by_text = [text.lower().split() for text in texts]
    vocabulary = sorted({
        token
        for tokens in tokens_by_text
        for token in tokens
    })
    vocabulary_ids = {token: index for index, token in enumerate(vocabulary)}
    embeddings = np.zeros((len(vocabulary), dimensions), dtype=np.float32)
    if not vocabulary or len(vocabulary) == 1:
        return vocabulary_ids, embeddings

    matrix = lil_matrix((len(vocabulary), len(vocabulary)), dtype=np.float32)
    for tokens in tokens_by_text:
        token_ids = [vocabulary_ids[token] for token in tokens]
        for index, token_id in enumerate(token_ids):
            start = max(0, index - window_size)
            end = min(len(token_ids), index + window_size + 1)
            for context_id in token_ids[start:index] + token_ids[index + 1:end]:
                matrix[token_id, context_id] += 1

    components = min(dimensions, max(1, len(vocabulary) - 1))
    embeddings[:, :components] = TruncatedSVD(
        n_components=components,
        random_state=42,
    ).fit_transform(matrix.tocsr())
    return vocabulary_ids, embeddings


def transform_static_embeddings(
    texts: list[str],
    vocabulary: dict[str, int],
    embeddings: np.ndarray,
) -> np.ndarray:
    vectors = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
    for row_index, text in enumerate(texts):
        token_ids = [
            vocabulary[token]
            for token in text.lower().split()
            if token in vocabulary
        ]
        if token_ids:
            vectors[row_index] = embeddings[token_ids].mean(axis=0)
    return vectors


def extract_representations(
    split: TextSplit,
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    static_dimensions: int = 100,
    batch_size: int = 1000,
    worker_count: int = 4,
) -> FeatureRepresentations:
    def extract_tfidf() -> tuple[Any, Any]:
        tfidf = TfidfVectorizer()
        return (
            tfidf.fit_transform(split["train_texts"]),
            tfidf.transform(split["test_texts"]),
        )

    def extract_static() -> tuple[np.ndarray, np.ndarray]:
        vocabulary, embeddings = fit_static_embeddings(
            split["train_texts"],
            static_dimensions,
        )
        return (
            transform_static_embeddings(split["train_texts"], vocabulary, embeddings),
            transform_static_embeddings(split["test_texts"], vocabulary, embeddings),
        )

    def extract_contextual() -> tuple[np.ndarray, np.ndarray]:
        from sentence_transformers import SentenceTransformer

        contextual = SentenceTransformer(sentence_model)
        return (
            contextual.encode(
                split["train_texts"],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            contextual.encode(
                split["test_texts"],
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        tfidf_future = executor.submit(extract_tfidf)
        static_future = executor.submit(extract_static)
        contextual_future = executor.submit(extract_contextual)

        tfidf_train, tfidf_test = tfidf_future.result()
        static_train, static_test = static_future.result()
        contextual_train, contextual_test = contextual_future.result()

    return {
        "split": split,
        "tfidf_train": tfidf_train,
        "tfidf_test": tfidf_test,
        "static_train": static_train,
        "static_test": static_test,
        "contextual_train": contextual_train,
        "contextual_test": contextual_test,
    }
