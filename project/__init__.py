from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import loader


def sine_func(x, A, omega, phi, c):
    return A * np.sin(omega * x + phi) + c


def best_fit_sine_regression(x, popt):
    return sine_func(x, *popt)


class Project:
    data: Dict[str, pd.DataFrame]

    def __init__(self, config: dict) -> None:
        self.config = config
        print(f"Climate Change v{self.config['version']}")

    def load(self) -> None:
        """Load data from csv files from the folder specified in the config."""
        self.data = loader.load(self.config["data"])

    def analyze_average_temperature(
        self, name: str, initial_days: int = 365 * 30
    ) -> None:
        """Analyze average temperature data and generate graphs."""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])
        df["TAVG"] = (df["TMIN"] + df["TMAX"]) / 2

        first_days = df.iloc[:initial_days]
        x = np.arange(len(first_days))
        y = first_days["TAVG"].values

        popt, _ = curve_fit(sine_func, x, y, p0=[1, 2 * np.pi / 365, 0, 0])

        df["YEAR"] = df["DATE"].dt.year
        grouped_data = df.groupby("YEAR")
        r_squared_values = []

        for year, group in grouped_data:
            x_year = np.arange(len(group))
            y_year = group["TAVG"].values
            r_squared = 1 - np.sum(
                (y_year - best_fit_sine_regression(x_year, popt)) ** 2
            ) / np.sum((y_year - np.mean(y_year)) ** 2)
            r_squared_values.append({"YEAR": year, "R-SQUARED": r_squared})

        r_squared_df = pd.DataFrame(r_squared_values)

        linear_fit = np.polyfit(r_squared_df["YEAR"], r_squared_df["R-SQUARED"], 1)
        linear_regression_eq = f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f}"

        plt.figure(figsize=(10, 6))
        plt.plot(
            r_squared_df["YEAR"],
            r_squared_df["R-SQUARED"],
            "o",
            label="R^2 vs Year",
        )
        plt.plot(
            r_squared_df["YEAR"],
            np.polyval(linear_fit, r_squared_df["YEAR"]),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("R^2")
        plt.title(f"R^2 vs Year for {name} (Average Temperature)")
        plt.text(
            0.01,
            0.98,
            linear_regression_eq,
            transform=plt.gca().transAxes,
            fontsize=12,
            va="top",
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}TEMP_{name}.png")
        plt.close()

    def analyze_monthly_precipitation(
        self, name: str, initial_months: int = 12 * 30
    ) -> None:
        """Analyze monthly precipitation data and generate graphs."""
        df = self.data.get(name)
        df = df.dropna(subset=["PRCP"])

        df["MONTH"] = df["DATE"].dt.month
        df["YEAR"] = df["DATE"].dt.year
        monthly_data = df.groupby(["YEAR", "MONTH"])["PRCP"].sum().reset_index()

        first_months = monthly_data.iloc[:initial_months]
        x = np.arange(len(first_months))
        y = first_months["PRCP"].values

        try:
            popt, _ = curve_fit(
                sine_func, x, y, p0=[1, 2 * np.pi / initial_months, 0, 0]
            )
        except RuntimeError:
            return

        years = monthly_data["YEAR"].unique()
        r_squared_values = []

        for year in years:
            group = monthly_data[monthly_data["YEAR"] == year]
            x_year = np.arange(len(group))
            y_year = group["PRCP"].values

            try:
                popt_year, _ = curve_fit(sine_func, x_year, y_year, p0=popt)
            except RuntimeError:
                continue

            y_predicted = sine_func(x_year, *popt_year)
            y_mean = np.mean(y_year)

            total_sum_of_squares = np.sum((y_year - y_mean) ** 2)
            residual_sum_of_squares = np.sum((y_year - y_predicted) ** 2)

            r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
            r_squared_values.append({"YEAR": year, "R-SQUARED": r_squared})

        r_squared_df = pd.DataFrame(r_squared_values)

        linear_fit = np.polyfit(r_squared_df["YEAR"], r_squared_df["R-SQUARED"], 1)
        linear_regression_eq = f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f}"

        plt.figure(figsize=(10, 6))
        plt.plot(
            r_squared_df["YEAR"],
            r_squared_df["R-SQUARED"],
            "o",
            label="R^2 vs Year",
        )
        plt.plot(
            r_squared_df["YEAR"],
            np.polyval(linear_fit, r_squared_df["YEAR"]),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("R^2")
        plt.title(f"R^2 vs Year for {name} (Monthly Precipitation)")
        plt.text(
            0.01,
            0.98,
            linear_regression_eq,
            transform=plt.gca().transAxes,
            fontsize=12,
            va="top",
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}PRCP_{name}.png")
        plt.close()
