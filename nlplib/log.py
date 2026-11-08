from pathlib import Path
from datetime import datetime
import sys
from nlplib.config import LOG_FILE

def clear_log(encoding: str | None = "utf-8") -> None:
    Path(LOG_FILE).parent.mkdir(parents = True, exist_ok = True)
    with open(LOG_FILE, "w", encoding = encoding) as f:
        f.write("")

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