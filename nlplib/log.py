from pathlib import Path
from nlplib.config import LOG_FILE

log_path = Path(__file__).resolve().parent.parent / LOG_FILE

def clear_log(encoding: str | None = "utf-8") -> None:
    with open(log_path, "w", encoding = encoding) as f:
        f.write("")

def write_log(text: str, encoding: str | None = "utf-8", skip_print: bool = False) -> None:
    with open(log_path, "a", encoding = encoding) as f:
        f.write(f"{str(text)}\n")
        if not skip_print:
            print(f"{str(text)}\n", end = "")