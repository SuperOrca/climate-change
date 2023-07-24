import numpy as np
import pandas as pd

from typing import Any

def calculate_average(row: pd.Series) -> np.floating[Any]:
    valid_values = [value for value in row if pd.notna(value)]
    return np.mean(valid_values) if valid_values else np.nan


def sine_function(x: np.ndarray, A: np.float64, omega: np.float64, phi: np.float64, C: np.float64) -> np.ndarray:
    return A * np.sin(omega * x + phi) + C


def split_dataframe(df: pd.DataFrame, chunk_size: int) -> Any:
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]


def calculate_r_squared(y_data: np.ndarray, best_fit_curve: np.ndarray) -> np.float64:
    residuals = y_data - best_fit_curve
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
