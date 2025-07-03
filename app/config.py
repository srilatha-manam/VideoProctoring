
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "app" / "logs"
LOG_DIR.mkdir(exist_ok=True)

# configure root logger
logger = logging.getLogger("proctoring")
logger.setLevel(logging.ERROR)  #  logs errors and critical issues

handler = RotatingFileHandler(
    LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
formatter = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
