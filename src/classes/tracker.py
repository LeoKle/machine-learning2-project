from collections import defaultdict
from typing import DefaultDict, List

DataDict = DefaultDict[str, List[float | int]]


class Tracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.last_tracked_epoch = None

    def track(self, metric: str, value: float, epoch: int):
        if epoch != self.last_tracked_epoch:
            self.metrics["epoch"].append(epoch)
            self.last_tracked_epoch = epoch
        self.metrics[metric].append(value)

    def get_metrics(self) -> DataDict:
        return self.metrics

    def import_data(self):
        pass

    def export_data(self):
        pass
