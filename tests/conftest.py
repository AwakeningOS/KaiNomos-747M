import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
