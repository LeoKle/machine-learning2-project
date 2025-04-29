from collections import defaultdict
from typing import DefaultDict, List

DataDict = DefaultDict[str, List[float | int]]


class Tracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def track(self, metric: str, value: float, epoch: int):
        self.metrics["epoch"].append(epoch)
        self.metrics[metric].append(value)

    def get_metrics(self) -> DataDict:
        return self.metrics

    def import_data(self):
        pass

    def export_data(self):
        pass
