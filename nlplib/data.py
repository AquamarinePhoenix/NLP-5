import csv
from nlplib.config import DATA_FILE

def read_raw_data(encoding: str | None = "utf-8") -> list[str]:
    with open(DATA_FILE, "r", newline = "", encoding = encoding) as f:
        reader = csv.DictReader(f)
        return list(reader)