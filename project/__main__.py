import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

from . import Project

with open("config.json", 'r') as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

df = project.data["stlouis_1938-2023.csv"].tail(1825)

def calculate_average(row):
    valid_values = [value for value in row if pd.notna(value)]
    return np.mean(valid_values) if valid_values else np.nan

df["AVG_TMAX_TMIN"] = df[["TMAX", "TMIN"]].apply(calculate_average, axis=1)

def sine_function(x, A, omega, phi, C):
    return A * np.sin(omega * x + phi) + C

x_data = np.arange(len(df))
y_data = df["AVG_TMAX_TMIN"].values

initial_guess = (np.max(y_data) - np.min(y_data)) / 2, 2 * np.pi / 365, 0, np.mean(y_data)

popt, _ = curve_fit(sine_function, x_data, y_data, p0=initial_guess)

best_fit_curve = sine_function(x_data, *popt)

residuals = y_data - best_fit_curve
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)

equation = f"y = {popt[0]:.2f} * sin({popt[1]:.6f} * x + {popt[2]:.2f}) + {popt[3]:.2f}"
r_squared_text = f"R-squared = {r_squared:.4f}"

print(equation)
print(r_squared_text)

plt.figure(figsize=(12, 8))
plt.plot(df["DATE"], df["AVG_TMAX_TMIN"], color="b", label="Average of TMAX and TMIN")
plt.plot(df["DATE"], best_fit_curve, color="r", label="Best Fit Sine Curve")
plt.xlabel("Date")
plt.ylabel("Average Temperature")
plt.title("Date vs Average Temperature")
plt.xticks(rotation=45)
plt.tight_layout()

plt.legend()
plt.savefig("assets/stlouis.png")
