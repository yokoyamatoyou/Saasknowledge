import os
from pathlib import Path


def load_dotenv(path=None):
    if not path:
        return True
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
    return True
