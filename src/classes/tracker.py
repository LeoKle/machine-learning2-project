from collections import defaultdict
import json
from math import nan
from pathlib import Path
from typing import DefaultDict, List

DataDict = DefaultDict[str, List[float | int]]


class Tracker:

    def __init__(self, output: Path):
        self.metrics = defaultdict(list)
        self.output = output

        self.__EPOCH_KEY = "epoch"

    def track(self, metric: str, value: float, epoch: int):
        # Ensure epoch list is extended up to `epoch`
        while len(self.metrics[self.__EPOCH_KEY]) < epoch:
            self.metrics[self.__EPOCH_KEY].append(
                len(self.metrics[self.__EPOCH_KEY]) + 1
            )

        # Determine how long all metric lists should be
        num_epochs = len(self.metrics[self.__EPOCH_KEY])

        # Ensure the specific metric list exists and is extended
        if len(self.metrics[metric]) < num_epochs:
            self.metrics[metric].extend(
                [nan] * (num_epochs - len(self.metrics[metric]))
            )

        # Extend all existing lists to match `num_epochs` if needed
        for key in self.metrics:
            if len(self.metrics[key]) < num_epochs:
                self.metrics[key].extend([nan] * (num_epochs - len(self.metrics[key])))

        # Set the value at the correct index
        self.metrics[metric][epoch - 1] = value

    def get_metrics(self) -> DataDict:
        return self.metrics

    def import_data(self):
        pass

    def export_data(self):
        with self.output.open("w") as f:
            json.dump(self.metrics, f, indent=2)
