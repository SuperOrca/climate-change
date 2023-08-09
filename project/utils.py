import numpy as np


def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c


def sine_func(x, A, omega, phi, c):
    return A * np.sin(omega * x + phi) + c


def best_fit_sine_regression(x, popt):
    return sine_func(x, *popt)
