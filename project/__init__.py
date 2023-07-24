from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import loader
from .utils import (
    calculate_average,
    calculate_r_squared,
    sine_function,
    split_dataframe,
)


class Project:
    data: Dict[str, pd.DataFrame]

    def __init__(self, CONFIG: dict) -> None:
        """Main project class."""
        self.CONFIG = CONFIG

        print(f"Climate Change v{self.CONFIG['version']}")

    def load(self) -> None:
        """Load data from csv files in folder specified in the config."""
        self.data = loader.load(self.CONFIG["data"]["path"])

    def compare_average_temperature_r_squared(self, name: str, years: int = 30) -> None:
        """Compare extremity of average temperature."""
        df = self.data[name]

        df["AVG_TMAX_TMIN"] = df[["TMAX", "TMIN"]].apply(calculate_average, axis=1)

        chunk_size = 365 * years
        x_data = np.arange(chunk_size)
        y_data = df["AVG_TMAX_TMIN"].iloc[:chunk_size].values

        valid_indices = np.isfinite(y_data)
        x_data_valid = x_data[valid_indices]
        y_data_valid = y_data[valid_indices]

        initial_guess = (
            (np.max(y_data_valid) - np.min(y_data_valid)) / 2,
            2 * np.pi / 365,
            0,
            np.mean(y_data_valid),
        )
        popt, _ = curve_fit(
            sine_function, x_data_valid, y_data_valid, p0=initial_guess, maxfev=5000
        )

        results = []
        for chunk_df in split_dataframe(df.iloc[chunk_size:], 365):
            x_data = np.arange(len(chunk_df))
            y_data = chunk_df["AVG_TMAX_TMIN"].values

            valid_indices = np.isfinite(y_data)
            x_data_valid = x_data[valid_indices]
            y_data_valid = y_data[valid_indices]

            best_fit_curve = sine_function(x_data_valid, *popt)
            r_squared = calculate_r_squared(y_data_valid, best_fit_curve)

            results.append({"Date": chunk_df["DATE"].iloc[-1], "R-squared": r_squared})

        results_df = pd.DataFrame(results)

        linear_coeffs = np.polyfit(
            np.arange(len(results_df)), results_df["R-squared"], 1
        )
        best_fit_line = np.polyval(linear_coeffs, np.arange(len(results_df)))

        print(
            f"{name}: Linear Equation: y = {linear_coeffs[0]:.4f} * x + {linear_coeffs[1]:.4f}"
        )

        linear_fit_residuals = results_df["R-squared"] - best_fit_line
        ss_res_linear = np.sum(linear_fit_residuals**2)
        ss_tot_linear = np.sum(
            (results_df["R-squared"] - np.mean(results_df["R-squared"])) ** 2
        )
        r_squared_linear = 1 - (ss_res_linear / ss_tot_linear)
        print(f"{name}: R-squared value of the linear fit: {r_squared_linear}")

        plt.figure(figsize=(12, 8))
        plt.plot(
            results_df["Date"], results_df["R-squared"], color="b", label="R-squared"
        )
        plt.plot(
            results_df["Date"],
            best_fit_line,
            color="r",
            linestyle="--",
            label="Best Fit Line",
        )
        plt.axhline(
            y=0, color="grey", linestyle="--", label="Zero R-squared (Baseline)"
        )
        plt.xlabel("Date")
        plt.ylabel("R-squared")
        plt.title("R-squared Values over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.legend()
        plt.savefig(self.CONFIG["output"]["path"] + name + ".png")
