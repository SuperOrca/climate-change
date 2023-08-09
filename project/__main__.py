import json

from . import Project

with open("config.json", "r") as file:
    CONFIG = json.load(file)

project = Project(CONFIG)
project.load()

name = "saltlakecity_1941-2022"

project.test_9(name, "test 9")
