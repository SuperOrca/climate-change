import os
from typing import Dict

import pandas as pd


def load(
    path: str, low_memory: bool = False, format: bool = True
) -> Dict[str, pd.DataFrame]:
    """Load data from csv files in data folder."""
    data = {}

    for file in os.listdir(path):
        df = pd.read_csv(path + file, low_memory=low_memory)

        if format:
            df = df[~df.apply(lambda row: row.astype(str).str.startswith("999")).any(axis=1)]
            df["DATE"] = pd.to_datetime(df["DATE"])

        data[file.rstrip(".csv")] = df

    return data
