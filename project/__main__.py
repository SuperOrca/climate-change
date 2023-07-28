import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

for name in project.data.keys():
    project.analyze_average_temperature(name)
    project.analyze_total_precipitation(name)
