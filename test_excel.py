

from pathlib import Path
import polars as pl

pq_path = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\reference_frames\marker_based\S08\00-static-stand.parquet"
)
df = pl.read_parquet(pq_path)
print("Segments:", df["segment"].unique().to_list())
print(df.group_by("segment").len())


