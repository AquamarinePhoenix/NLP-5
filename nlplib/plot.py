import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import Counter, defaultdict
from pathlib import Path
from wordcloud import WordCloud
from nlplib.config import COLORS

THEME_BACKGROUND = "#f2f2f2"
SHADOW_EFFECTS = [
    pe.SimpleLineShadow(offset=(1, -1), shadow_color="grey", alpha=0.3),
    pe.Normal(),
]


def _style_axes(figure, axes) -> None:
    figure.set_facecolor(THEME_BACKGROUND)
    axes.set_facecolor(THEME_BACKGROUND)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)


def _style_title(title):
    title.set_path_effects(SHADOW_EFFECTS)


def plot_line_chart(
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str,
    rows: list[str, str],
    show: bool = False,
    save: bool = True,
):
    figure = plt.figure(figsize=(10, 6))
    axes = figure.gca()
    _style_axes(figure, axes)

    for index, emotion_label in enumerate(sorted({row[0] for row in rows})):
        points = [row for row in rows if row[0] == emotion_label]
        line, = axes.plot(
            [row[1] for row in points],
            [row[2] for row in points],
            label=emotion_label,
            color=COLORS[index % len(COLORS)],
        )
        line.set_path_effects(SHADOW_EFFECTS)
        
    _style_title(axes.set_title(title))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(title=legend_title)
    plt.tight_layout()
    if save:
        output_path = Path("data/output")
        output_path.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path / "text_length_distribution.png", dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def plot_emotion_word_clouds(
    rows: list[tuple[str, str]],
    output_directory: str = "data/output/wordclouds",
    top_words: int = 5,
    show: bool = False,
    save: bool = True,
) -> dict[str, list[tuple[str, int]]]:
    words_by_emotion: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for emotion, transformed_text in rows:
        words_by_emotion[emotion].update(transformed_text.split())

    output_path = Path(output_directory)
    if save:
        output_path.mkdir(parents=True, exist_ok=True)
    frequent_words: dict[str, list[tuple[str, int]]] = {}

    for index, emotion in enumerate(sorted(words_by_emotion)):
        frequencies = words_by_emotion[emotion]
        frequent_words[emotion] = frequencies.most_common(top_words)
        word_cloud = WordCloud(
            width=1200,
            height=800,
            background_color=THEME_BACKGROUND,
            colormap="viridis",
        ).generate_from_frequencies(frequencies)

        figure, axes = plt.subplots(figsize=(12, 8))
        _style_axes(figure, axes)
        axes.imshow(word_cloud, interpolation="bilinear")
        axes.axis("off")
        _style_title(axes.set_title(f"Most frequent words: {emotion}"))
        figure.tight_layout()
        if save:
            figure.savefig(
                output_path / f"wordcloud_{emotion}.png",
                dpi=150,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(figure)

    return frequent_words


def plot_pos_distribution(
    rows: list[tuple[str, str, int, float]],
    output_file: str = "data/output/pos_distribution.png",
    show: bool = False,
    save: bool = True,
) -> None:
    emotions = sorted({row[0] for row in rows})
    pos_tags = sorted({row[1] for row in rows})
    if not emotions or not pos_tags:
        return

    positions = list(range(len(pos_tags)))
    width = 0.8 / len(emotions)
    figure, axes = plt.subplots(figsize=(12, 6))
    _style_axes(figure, axes)

    for index, emotion in enumerate(emotions):
        percentages = {
            pos: percentage
            for row_emotion, pos, _, percentage in rows
            if row_emotion == emotion
        }
        offsets = [
            position - 0.4 + width / 2 + index * width
            for position in positions
        ]
        bars = axes.bar(
            offsets,
            [percentages.get(pos, 0) for pos in pos_tags],
            width=width,
            label=emotion,
            color=COLORS[index % len(COLORS)],
        )
        for bar in bars:
            bar.set_path_effects(SHADOW_EFFECTS)

    _style_title(axes.set_title("Part-of-speech distribution by emotion"))
    axes.set_xlabel("Part-of-speech tag")
    axes.set_ylabel("Percentage of tokens")
    axes.set_xticks(positions, pos_tags)
    axes.legend(title="Emotion class")
    figure.tight_layout()
    if save:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)