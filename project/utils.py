import numpy as np


def sine_func(x, A, omega, phi, c):
    return A * np.sin(omega * x + phi) + c


def best_fit_sine_regression(x, popt):
    return sine_func(x, *popt)
