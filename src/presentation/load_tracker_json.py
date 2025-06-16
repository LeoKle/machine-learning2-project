import json
from pathlib import Path

json_path = Path(
    "output/gan_IS_FID_cifar10_generatorcnn2_discriminatorcnn/0/gan_metrics.json"
)

with json_path.open("r") as f:
    data = json.load(f)

print(data)
