import numpy as np
from scipy.stats import anderson_ksamp, kurtosis, skew

from nlplib.log import write_log


def log_text_length_statistics(distribution: list[tuple[str, int, int]]) -> None:
    samples_by_emotion: dict[str, list[tuple[int, int]]] = {}

    for emotion, text_length, record_count in distribution:
        samples_by_emotion.setdefault(emotion, []).append(
            (text_length, record_count)
        )

    samples = {}
    for emotion, length_counts in samples_by_emotion.items():
        lengths = np.array([length for length, _ in length_counts], dtype=float)
        counts = np.array([count for _, count in length_counts], dtype=int)
        sample = np.repeat(lengths, counts)
        samples[emotion] = sample
        write_log(
            f"Text length statistics for {emotion}: "
            f"n={sample.size}, "
            f"skewness={skew(sample, bias=False):.6f}, "
            f"excess_kurtosis={kurtosis(sample, fisher=True, bias=False):.6f}"
        )

    if len(samples) < 2:
        write_log(
            "Anderson-Darling k-sample test skipped: "
            "at least two emotion distributions are required."
        )
        return

    result = anderson_ksamp(list(samples.values()), variant="midrank")
    write_log(
        "Anderson-Darling k-sample test for text length by emotion: "
        f"statistic={result.statistic:.6f}, "
        f"p_value={result.pvalue:.6g}"
    )
