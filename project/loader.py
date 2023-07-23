import os
from typing import Dict

import pandas as pd


def load(path: str, low_memory: bool = False, format: bool = True) -> Dict[str, pd.DataFrame]:
    """Load data from csv files in data folder."""
    data = {}

    for file in os.listdir(path):
        df = pd.read_csv(path + file, low_memory=low_memory)

        if format:
            df["DATE"] = pd.to_datetime(df["DATE"])

        data[file] = df

    return data
