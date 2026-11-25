from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

def load_data():
    """Load the course catalogue and student interaction records."""
    courses_path = DATA_DIR / "courses.csv"
    interactions_path = DATA_DIR / "interactions.csv"
    if not courses_path.exists() or not interactions_path.exists():
        raise FileNotFoundError("courses.csv and interactions.csv must be in the project folder.")
    return pd.read_csv(courses_path), pd.read_csv(interactions_path)
