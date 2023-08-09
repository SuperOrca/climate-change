import os
import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "miami_1948-2022"

project.test_9(name, "TMAX", "John Glenn Columbus International Airport: Daily High")
