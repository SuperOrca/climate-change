from typing import Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .loader import load
from .utils import quadratic_function, sine_func, best_fit_sine_regression


class Project:
    data: Dict[str, pd.DataFrame]

    def __init__(self, config: dict) -> None:
        self.config = config
        print(f"Climate Change v{self.config['version']}")

    def load(self) -> None:
        """Load data from csv files from the folder specified in the config."""
        self.data = load(self.config["data"])

    def test_9(self, name: str, title: str):
        df = self.data.get(name).dropna(subset=["TMIN", "TMAX"])
        # df["TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)
        df["DOY"] = df["DATE"].dt.day_of_year

        daily_stats = df.groupby("DOY")["TMAX"].agg(["mean", "std"]).reset_index()

        excursions = {"YEAR": [], "EXCUR2": [], "EXCUR3": []}

        for year, group in df.groupby(df["DATE"].dt.year):
            excursions["YEAR"].append(year)
            value2 = 0
            value3 = 0
            for _, day in group.iterrows():
                day_stats = daily_stats[daily_stats["DOY"] == day["DOY"]]
                mean = day_stats["mean"].values[0]
                std = day_stats["std"].values[0]
                value = day["TMAX"]

                if value >= mean + std * 2:
                    value2 += 1
                # if value < mean - std * 2:
                #     value2 += 1

                if value >= mean + std * 3:
                    value3 += 1
            excursions["EXCUR2"].append(value2)
            excursions["EXCUR3"].append(value3)

        edf = pd.DataFrame(excursions)
        edf.to_csv("excursion.csv")

        plt.figure(figsize=(10, 6))
        plt.plot(edf["YEAR"], edf["EXCUR2"], "o")
        plt.xlabel("Year")
        plt.ylabel("# of High Temperature Excursions (2 sigma)")
        plt.title(f"{title} (2 sigma)")
        plt.savefig(f"{self.config['output']}{name}_2.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(edf["YEAR"], edf["EXCUR3"], "o")
        plt.xlabel("Year")
        plt.ylabel("# of High Temperature Excursions (3 sigma)")
        plt.title(f"{title} (3 sigma)")
        plt.savefig(f"{self.config['output']}{name}_3.png")
        plt.close()

    def test_8(self, name: str, title: str, initial: int = 30):
        """graph year vs standard deviation"""
        df = self.data.get(name).dropna(subset=["TMIN", "TMAX"])
        # df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)
        df.loc[:, "YEAR"] = df["DATE"].dt.year

        stds = df.groupby("YEAR")["TMAX"].std()

        avg_std = stds.head(initial).mean()

        stds = stds - avg_std

        linear_fit = np.polyfit(stds.index, stds.values, 1)

        y_pred = np.polyval(linear_fit, stds.index)
        mean_y = np.mean(stds.values)
        tss = np.sum((stds.values - mean_y) ** 2)
        rss = np.sum((stds.values - y_pred) ** 2)
        r_squared = 1 - (rss / tss)

        linear_regression_eq = (
            f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f} (R^2 = {r_squared:.4f})"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            stds.index,
            stds.values,
            "o",
            label="Data",
        )
        plt.plot(
            stds.index,
            np.polyval(linear_fit, stds.index),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("Difference in Standard Deviation")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}{name}.png")
        plt.close()

    def test_7(self, name: str, title: str, initial: int = 365 * 30):
        """graph year vs r^2 using quadratic fit"""
        df = self.data.get(name).dropna(subset=["TMIN", "TMAX"])
        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)
        df.loc[:, "DAY"] = df["DATE"].dt.dayofyear / df["DATE"].dt.is_leap_year.apply(
            lambda x: 366 if x else 365
        )

        idf = df.loc[:initial]

        x = idf["DAY"]
        y = idf["TAVG"]
        A = np.vstack([x**2, x, np.ones_like(x)]).T
        coefficients, _ = np.linalg.lstsq(A, y, rcond=None)[:2]

        years = df["DATE"].dt.year.unique()
        ydf_data = {"YEAR": [], "R^2": []}

        for year in years:
            ydf_data["YEAR"].append(year)
            ydf_data["R^2"].append(self.helper_7(df, coefficients, year))

        ydf = pd.DataFrame(ydf_data)

        linear_fit = np.polyfit(ydf["YEAR"], ydf["R^2"], 1)

        y_pred = np.polyval(linear_fit, ydf["YEAR"])
        mean_y = np.mean(ydf["R^2"])
        tss = np.sum((ydf["R^2"] - mean_y) ** 2)
        rss = np.sum((ydf["R^2"] - y_pred) ** 2)
        r_squared = 1 - (rss / tss)

        linear_regression_eq = (
            f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f} (R^2 = {r_squared:.4f})"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            ydf["YEAR"],
            ydf["R^2"],
            "o",
            label="Data",
        )
        plt.plot(
            ydf["YEAR"],
            np.polyval(linear_fit, ydf["YEAR"]),
            "-",
            label="Linear Fit",
        )
        plt.xlabel("Year")
        plt.ylabel("R-Squared")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}{name}.png")
        plt.close()

    def helper_7(self, df: pd.DataFrame, coefficients: list, year: int):
        """helper method for test_7"""
        ydf = df[df["DATE"].dt.year == year]

        mean_y = np.mean(ydf["TAVG"])
        SS_total = np.sum((ydf["TAVG"] - mean_y) ** 2)
        SS_residual = np.sum((ydf["TAVG"] - np.polyval(coefficients, ydf["DAY"])) ** 2)
        R_squared = 1 - SS_residual / SS_total

        return R_squared

    def test_6(self, name: str, title: str, initial_days: int = 365 * 30) -> None:
        """graph year vs average"""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        first_days = df.iloc[:initial_days].reset_index()
        x = np.arange(len(first_days))
        y = first_days["TAVG"].values

        popt, _ = curve_fit(sine_func, x, y, p0=[1, 2 * np.pi / 365, 0, 0])

        x_fit = pd.date_range(start=df["DATE"].min(), periods=len(df), freq="D")
        y_fit = sine_func(np.arange(len(x_fit)), *popt)

        plt.figure(figsize=(60, 8))
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

    def test_5(self, name: str, title: str, initial_days: int = 365 * 30) -> None:
        """graph year vs r^2 using sine prediction"""
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

    def test_4(self, name: str, title: str) -> None:
        """graph year vs maximum annual precipitation"""
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

    def test_3(
        self, name: str, title: str, year: int, initial_days: int = 365 * 30
    ) -> None:
        """graph date vs average temperature of specific year with sine prediction"""
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

    def test_2(self, name: str, title: str) -> None:
        """graph year vs average temperature on january 1st"""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        df = df[df["DATE"].dt.day_of_year == 1]

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

    def test_1(self, name: str, title: str) -> None:
        """graph year vs average annual temperature"""
        df = self.data.get(name)
        df = df.dropna(subset=["TMIN", "TMAX"])

        df.loc[:, "TAVG"] = df[["TMIN", "TMAX"]].mean(axis=1)

        df["YEAR"] = df["DATE"].dt.year
        yearly_data = df.groupby(["YEAR"])["TAVG"].mean().reset_index()

        linear_fit = np.polyfit(yearly_data["YEAR"], yearly_data["TAVG"], 1)
        linear_regression_eq = f"y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f}"

        plt.figure(figsize=(10, 6))
        plt.plot(
            yearly_data["YEAR"],
            yearly_data["TAVG"],
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
        plt.ylabel("Average Temperature (°C)")
        plt.title(title)
        plt.annotate(
            linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        )
        plt.legend()

        plt.savefig(f"{self.config['output']}TAVG_{name}.png")
        plt.close()
