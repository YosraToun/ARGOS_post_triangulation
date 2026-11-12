from pathlib import Path

import polars as pl
import pandas as pd




def generate_excel_summary(df: pl.DataFrame, output_path: Path):
    """
    Generate an Excel summary of kinematic errors from a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing kinematic error data with columns:
            - "subject": Subject identifier
            - "task": Task identifier
            - "model": Model identifier
            - "dof": Degree of freedom identifier
            - "error_deg": Kinematic error in degrees
            - "abs_error_deg": Absolute kinematic error in degrees
    output_path : Path
        Path to the output Excel file where the summary will be saved.
    """

    statistics_abs = [
        pl.len().alias("n_frames"),
        pl.col("abs_error_deg").mean().alias("mean_abs_error"),
        pl.col("abs_error_deg").median().alias("median_abs_error"),
        pl.col("abs_error_deg").std().alias("std_abs_error"),
        pl.col("abs_error_deg").quantile(0.25).alias("q25"),
        pl.col("abs_error_deg").quantile(0.75).alias("q75"),
        pl.col("abs_error_deg").quantile(0.95).alias("p95"),
        pl.col("abs_error_deg").max().alias("max_abs_error"),
    ]
    statistics_err = [
        pl.len().alias("n_frames"),
        pl.col("error_deg").mean().alias("mean_error"),
        pl.col("error_deg").median().alias("median_error"),
        pl.col("error_deg").std().alias("std_error"),
        pl.col("error_deg").quantile(0.25).alias("q25"),
        pl.col("error_deg").quantile(0.75).alias("q75"),
        pl.col("error_deg").quantile(0.95).alias("p95"),
        pl.col("error_deg").max().alias("max_error"),
    ]

    summary_by_subject_abs = (
        df.group_by(["subject", "task", "model", "dof"])
        .agg(statistics_abs)
        .sort(["subject", "task", "dof", "model"])
    )
    summary_by_task_dof_abs = (
        df.group_by(["task", "dof", "model"])
        .agg(statistics_abs)
        .sort(["task", "dof", "model"])
    )
    summary_by_subject_err = (
        df.group_by(["subject", "task", "model", "dof"])
        .agg(statistics_err)
        .sort(["subject", "task", "dof", "model"])
    )
    summary_by_task_dof_err = (
        df.group_by(["task", "dof", "model"])
        .agg(statistics_err)
        .sort(["task", "dof", "model"])
    )

    with pd.ExcelWriter(output_path) as writer:
        summary_by_subject_abs.to_pandas().to_excel(
            writer,
            sheet_name="By_subject",
            index=False,
        )
        summary_by_task_dof_abs.to_pandas().to_excel(
            writer,
            sheet_name="By_task_DOF",
            index=False,
        )
        summary_by_subject_err.to_pandas().to_excel(
            writer,
            sheet_name="By_subject_Error",
            index=False,
        )
        summary_by_task_dof_err.to_pandas().to_excel(
            writer,
            sheet_name="By_task_DOF_Error",
            index=False,
        )

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    
    df = pl.read_parquet("kinematic_errors_strong_side.parquet")
    output_path = Path("kinematic_errors_summary.xlsx")
    generate_excel_summary(df, output_path)