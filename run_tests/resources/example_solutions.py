from pathlib import Path

import pandas as pd


file_dir = Path(__file__).parent

with open(file_dir / "solutions.csv", "r") as fp:
    solutions = pd.read_csv(fp, header=0, index_col=0)
