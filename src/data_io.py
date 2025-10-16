from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

def load_csv(filename: str) -> pd.DataFrame:
    """
    Load CSV from Data/ folder. Example: load_csv("Predict Hair Fall.csv")
    Does not mutate dataframe (returns a copy).
    """
    path = DATA_DIR / filename
    return pd.read_csv(path)

def save_df(df: pd.DataFrame, filename: str) -> None:
    """Save df to Data/ or artifacts/ (choose path)."""
    df.to_csv(filename, index=False)
