import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "tm5_manual_book_grasp" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
