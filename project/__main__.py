import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "miami_1948-2023"

project.analyze_average_temperature(name, "tavg")
project.analyze_maximum_precipitation(name, "prcp")