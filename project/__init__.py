from typing import Dict

import pandas as pd

from . import loader


class Project:
    data: Dict[str, pd.DataFrame]

    def __init__(self, config: dict) -> None:
        self.config = config

    def load(self) -> None:
        self.data = loader.load(self.config["data"]["path"])
