import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "indianapolis_1942-2023"

# project.analyze_average_temperature(name, "Indianapolis International Airport (Average Temperature)")
# project.analyze_yearly_precipitation(name, "Indianapolis International Airport")
project.test(name, "Indianapolis International Airport (Average Temperature)", 1943)