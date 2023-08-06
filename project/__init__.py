from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .loader import load


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
        self.data = load(self.config["data"])

    def full_average_temperature(
        self, name: str, title: str, initial_days: int = 365 * 30
    ) -> None:
        """Analyze average temperature data and generate graphs."""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        first_days = df.iloc[:initial_days]
        x = np.arange(len(first_days))
        y = first_days["TAVG"].values

        popt, _ = curve_fit(sine_func, x, y, p0=[1, 2 * np.pi / 365, 0, 0])

        x_fit = pd.date_range(start=df["DATE"].min(), periods=len(df), freq="D")
        y_fit = sine_func(np.arange(len(x_fit)), *popt)

        plt.figure(figsize=(60, 6))
        plt.plot(
            df["DATE"],
            df["TAVG"],
            "o",
            label="Data",
        )
        plt.plot(
            x_fit,
            y_fit,
            "-",
            label="Sine Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("Average Temperature")
        plt.title(title)
        plt.legend()

        plt.savefig(f"{self.config['output']}FULL_TAVG_{name}.png")
        plt.close()

    def analyze_average_temperature(
        self, name: str, title: str, initial_days: int = 365 * 30
    ) -> None:
        """Analyze average temperature data and generate graphs."""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

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

        y_pred = np.polyval(linear_fit, r_squared_df["YEAR"])
        mean_y = np.mean(r_squared_df["R-SQUARED"])
        tss = np.sum((r_squared_df["R-SQUARED"] - mean_y) ** 2)
        rss = np.sum((r_squared_df["R-SQUARED"] - y_pred) ** 2)
        r_squared = 1 - (rss / tss)

        linear_regression_eq = (
            f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f} (R^2 = {r_squared:.4f})"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            r_squared_df["YEAR"],
            r_squared_df["R-SQUARED"],
            "o",
            label="Data",
        )
        plt.plot(
            r_squared_df["YEAR"],
            np.polyval(linear_fit, r_squared_df["YEAR"]),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("R^2")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}TAVG_{name}.png")
        plt.close()

    def analyze_maximum_precipitation(self, name: str, title: str) -> None:
        """Analyze monthly precipitation data and generate graphs."""
        df = self.data.get(name)
        df = df.dropna(subset=["PRCP"])

        df["YEAR"] = df["DATE"].dt.year
        yearly_data = df.groupby(["YEAR"])["PRCP"].max().reset_index()

        linear_fit = np.polyfit(yearly_data["YEAR"], yearly_data["PRCP"], 1)
        linear_regression_eq = f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f}"

        plt.figure(figsize=(10, 6))
        plt.plot(
            yearly_data["YEAR"],
            yearly_data["PRCP"],
            "o",
            label=f"Data",
        )
        plt.plot(
            yearly_data["YEAR"],
            np.polyval(linear_fit, yearly_data["YEAR"]),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("Maximum Precipitation (mm)")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}MAX_PRCP_{name}.png")
        plt.close()

    def analyze_annual_average_temperature(
        self, name: str, title: str, year: int, initial_days: int = 365 * 30
    ) -> None:
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        first_days = df.iloc[:initial_days]
        x = np.arange(len(first_days))
        y = first_days["TAVG"].values

        popt, _ = curve_fit(sine_func, x, y, p0=[1, 2 * np.pi / 365, 0, 0])

        df = df[df["DATE"].dt.year == year]

        x_range = pd.date_range(start=df["DATE"].min(), periods=len(df), freq="D")
        y_fit = best_fit_sine_regression(np.arange(len(x_range)), popt)

        avg, _ = curve_fit(
            sine_func,
            np.arange(len(df)),
            df["TAVG"].values,
            p0=[1, 2 * np.pi / 365, 0, 0],
        )
        avg_fit = best_fit_sine_regression(np.arange(len(x_range)), avg)

        plt.figure(figsize=(10, 6))
        plt.plot(
            df["DATE"],
            df["TAVG"],
            "o",
            label="Data",
        )
        plt.plot(
            x_range,
            y_fit,
            "-",
            label="30 Year Average",
        )
        plt.plot(
            x_range,
            avg_fit,
            "-",
            label="Year Average",
        )
        plt.xlabel("Date")
        plt.ylabel("Average Temperature (°C)")
        plt.title(title)
        plt.legend()

        plt.savefig(f"{self.config['output']}TAVG_{name}_{year}.png")
        plt.close()

    def analyze_single_annual_average_temperature(self, name: str, title: str) -> None:
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        df = df[df["DATE"].dt.month == 1]
        df = df[df["DATE"].dt.day == 1]

        linear_fit = np.polyfit(df["DATE"].dt.year, df["TAVG"], 1)

        y_pred = np.polyval(linear_fit, df["DATE"].dt.year)
        mean_y = np.mean(df["TAVG"])
        tss = np.sum((df["TAVG"] - mean_y) ** 2)
        rss = np.sum((df["TAVG"] - y_pred) ** 2)
        r_squared = 1 - (rss / tss)

        linear_regression_eq = (
            f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f} (R^2 = {r_squared:.4f})"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            df["DATE"].dt.year,
            df["TAVG"],
            "o",
            label="Data",
        )
        plt.plot(
            df["DATE"].dt.year,
            np.polyval(linear_fit, df["DATE"].dt.year),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("January 1st Average Temperature (°C)")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}STAVG_{name}.png")
        plt.close()
