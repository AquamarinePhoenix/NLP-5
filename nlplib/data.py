from nlplib.config import DATA_FILE

EMOTIONS = [
    {"emotion_label": "D", "emotion_definition": "Džiaugsmas"},
    {"emotion_label": "L", "emotion_definition": "Liudesys"},
    {"emotion_label": "N", "emotion_definition": "Nuostaba"},
    {"emotion_label": "P", "emotion_definition": "Pyktis"},
    {"emotion_label": "Z", "emotion_definition": "Neutrali"}
]

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