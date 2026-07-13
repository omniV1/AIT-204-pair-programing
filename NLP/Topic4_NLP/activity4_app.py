"""Streamlit Cloud entry point. The app implementation lives in src/."""
from pathlib import Path
import runpy
import sys

src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))

runpy.run_path(str(src_dir / "activity4_app.py"), run_name="__main__")
