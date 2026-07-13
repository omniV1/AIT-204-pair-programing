"""Streamlit Cloud entry point. The app implementation lives in src/."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "src" / "activity4_app.py"), run_name="__main__")
