import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

for name in project.data.keys():
    project.compare_average_temperature_r_squared(name)
