import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "indianapolis_1942-2023"

# project.analyze_average_temperature(name, "Indianapolis International Airport (Average Temperature)")
# project.analyze_yearly_precipitation(name, "Indianapolis International Airport")
for i in range(2015, 2024):
    try:
        project.test(
            name, "Indianapolis International Airport (Average Temperature)", i
        )
    except:
        pass

for i in range(1942, 1951):
    try:
        project.test(
            name, "Indianapolis International Airport (Average Temperature)", i
        )
    except:
        pass
