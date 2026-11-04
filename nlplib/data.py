from nlplib.config import DATA_FILE

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