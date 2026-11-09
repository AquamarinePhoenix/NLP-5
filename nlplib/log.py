from pathlib import Path
from datetime import datetime
import sys
from nlplib.config import LOG_FILE

def clear_log(encoding: str | None = "utf-8") -> None:
    Path(LOG_FILE).parent.mkdir(parents = True, exist_ok = True)
    with open(LOG_FILE, "w", encoding = encoding) as f:
        f.write("")


def clear_output_pngs() -> None:
    output_directory = Path("data/output")
    if not output_directory.exists():
        return

    for output_file in output_directory.rglob("*"):
        if output_file.is_file() and output_file.suffix.lower() == ".png":
            output_file.unlink()


def write_log(text: str, encoding: str | None = "utf-8", skip_print: bool = False) -> None:
    Path(LOG_FILE).parent.mkdir(parents = True, exist_ok = True)
    with open(LOG_FILE, "a", encoding = encoding) as f:
        timestamp = datetime.now().isoformat()
        message = f"[{timestamp}] {str(text)}"
        f.write(f"{message}\n")
        if not skip_print:
            encoding = sys.stdout.encoding or "utf-8"
            safe_message = message.encode(encoding, errors="replace").decode(encoding)
            print(safe_message, end="\n\n")