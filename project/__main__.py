import os
import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

project.test_2(
    "ithaca_1893-2022",
    "Cornell University (Average Temperature on January 1st)",
)
