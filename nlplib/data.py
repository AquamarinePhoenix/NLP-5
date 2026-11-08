from nlplib.config import DATA_FILE, TEST, TEST_ROWS
import random

def read_raw_data(encoding: str | None = "utf-8") -> list[dict[str, str]]:
    with open(DATA_FILE, "r", encoding=encoding) as f:
        return [
            {
                "raw_input": line.rstrip("\r\n"),
                "emotion_label": line.split("/", 2)[1],
                "text_sample": line.rsplit("|", 1)[1].strip().rstrip("\\").lower()
            }
            for line in f
            if line.strip()
        ]

def prepare_data():

    data = read_raw_data()

    if TEST:
        classes = sorted({row["emotion_label"] for row in data})

        if len(classes) >= 2:
            selected_classes = random.Random(42).sample(classes, 2)
            rows_per_class = TEST_ROWS // 2
            selected_data = []

            for emotion_class in selected_classes:
                class_rows = [
                    row for row in data
                    if row["emotion_label"] == emotion_class
                ]
                selected_data.extend(class_rows[:rows_per_class])

            if TEST_ROWS % 2:
                remainder_rows = [
                    row
                    for row in data
                    if row["emotion_label"] == selected_classes[0]
                ][rows_per_class:rows_per_class + 1]
                selected_data.extend(remainder_rows)

            data = selected_data

    return data