import pandas as pd

# Load the parquet file
df = pd.read_parquet(r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\workspace\ARGOS_post_triangulation\data\pre_processed\Robeson_3Dtrajectories_errors.parquet")

# View the first few rows
print(df.head())

# View column names, data types, and non-null counts
print(df.info())

# View basic statistical summaries
print(df.describe())