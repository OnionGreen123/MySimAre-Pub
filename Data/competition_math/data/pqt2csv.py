import pandas as pd
import csv

df = pd.read_parquet("train-00000-of-00001-7320a6f3aba8ebd2.parquet")

# Safer CSV: quote everything so multiline fields stay valid
df.to_csv(
    "train_fixed.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    escapechar="\\",
    lineterminator="\n",
)