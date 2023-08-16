from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loader import load


class Project:
    data: Dict[str, pd.DataFrame]

    def __init__(self, config: dict) -> None:
        self.config = config
        print(f"Climate Change v{self.config['version']}")

    def load(self) -> None:
        """Load data from csv files from the folder specified in the config."""
        self.data = load(self.config["data"])

    def method_3(self, name: str, title: str):
        """Comparing number of statistically extreme daily high temperature events per year."""
        df = self.data.get(name).dropna(subset=["TMIN", "TMAX"])
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

    def method_2(self, name: str, title: str, initial: int = 365 * 30):
        """Comparing historical average temperature fit to individual years."""
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
            ydf_data["R^2"].append(self.helper_2(df, coefficients, year))

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

    def helper_2(self, df: pd.DataFrame, coefficients: list, year: int):
        """Helper for method_2. Not for testing."""
        ydf = df[df["DATE"].dt.year == year]

        mean_y = np.mean(ydf["TAVG"])
        SS_total = np.sum((ydf["TAVG"] - mean_y) ** 2)
        SS_residual = np.sum((ydf["TAVG"] - np.polyval(coefficients, ydf["DAY"])) ** 2)
        R_squared = 1 - SS_residual / SS_total

        # if year == 1948 or year == 2020:
        #     linear_regression_eq = f"y = {coefficients[0]:.4f}x^2 + {coefficients[1]:.4f}x + {coefficients[2]:.4f} (R^2 = {R_squared:.4f})"
            
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(
        #         ydf["DAY"],
        #         ydf["TAVG"],
        #         "o",
        #         label="Data",
        #     )
        #     plt.plot(
        #         ydf["DAY"],
        #         quadratic_function(ydf["DAY"], *coefficients),
        #         "-",
        #         label="Quadratic Fit",
        #     )
        #     plt.xlabel("Normalized Year")
        #     plt.ylabel("Average Temperature (°C)")
        #     plt.title(f"John Glenn Columbus International Airport in {year} (Average Temperature Quradratic Prediction)")
        #     plt.annotate(
        #         linear_regression_eq, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=12
        #     )
        #     plt.legend()

        #     plt.savefig(f"{self.config['output']}test_{year}.png")
        #     plt.close()

        return R_squared

    def method_1(self, name: str, title: str) -> None:
        """Comparing number of statistically extreme daily high temperature events per year."""
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
