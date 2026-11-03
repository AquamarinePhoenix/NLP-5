from nlplib.config import LOG_FILE

def clear_log(encoding: str | None = "utf-8") -> None:
    with open(LOG_FILE, "w", encoding = encoding) as f:
        f.write("")

def write_log(text: str, encoding: str | None = "utf-8", skip_print: bool = False) -> None:
    with open(LOG_FILE, "a", encoding = encoding) as f:
        f.write(f"{str(text)}\n")
        if not skip_print:
            print(f"{str(text)}\n", end = "")