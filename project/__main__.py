import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "minneapolis_1938-2022"

project.analysis(name, "test", initial=365 * 20)
