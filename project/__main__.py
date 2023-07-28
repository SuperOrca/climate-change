import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

YEARS = 25

for name in project.data.keys():
    project.analyze_average_temperature(name, 365 * YEARS)
    project.analyze_monthly_precipitation(name, 12 * YEARS)
