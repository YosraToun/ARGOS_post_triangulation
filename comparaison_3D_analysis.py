import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_error_quantiles(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    """
    Plot error distributions using:
        whisker low  = 5th percentile
        box low      = 20th percentile
        median       = 50th percentile
        box high     = 70th percentile
        whisker high = 95th percentile

    subjects and tasks determine which data are integrated
    into the distributions.
    """

    # ---------------------------------------------------------
    # Filter data
    # ---------------------------------------------------------

    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    # ---------------------------------------------------------
    # Quantiles
    # ---------------------------------------------------------

    error_columns = [
        "error_x",
        "error_y",
        "error_z",
        "error_norm",
    ]

    quantiles = (
        df_plot.unpivot(
            index=["point_short", "model"],
            on=error_columns,
            variable_name="component",
            value_name="error",
        )
        .drop_nulls("error")
        .group_by(["point_short", "model", "component"])
        .agg(
            pl.col("error").quantile(0.05).alias("q05"),
            pl.col("error").quantile(0.20).alias("q20"),
            pl.col("error").quantile(0.50).alias("median"),
            pl.col("error").quantile(0.70).alias("q70"),
            pl.col("error").quantile(0.95).alias("q95"),
            pl.len().alias("n"),
        )
        .sort(["point_short", "model", "component"])
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "X error",
            "Y error",
            "Z error",
            "Error norm",
        ],
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    subplot_positions = {
        "error_x": (1, 1),
        "error_y": (1, 2),
        "error_z": (2, 1),
        "error_norm": (2, 2),
    }

    # Consistent color for each model
    model_colors = {
        "SynthPose": "#636EFA",
        "RTMPose": "#EF3B3B",
    }

    model_list = df_plot.select("model").unique().sort("model").to_series().to_list()

    point_list = df_plot.select("point_short").unique().sort("point_short").to_series().to_list()

    # ---------------------------------------------------------
    # Add traces
    # ---------------------------------------------------------

    for component, (row, col) in subplot_positions.items():

        for model in model_list:

            data = quantiles.filter((pl.col("component") == component) & (pl.col("model") == model))

            if data.is_empty():
                continue

            # Keep same point order on all subplots
            data = pl.DataFrame({"point_short": point_list}).join(
                data,
                on="point_short",
                how="left",
            )

            x = data["point_short"].to_list()

            q05 = data["q05"].to_list()
            q20 = data["q20"].to_list()
            median = data["median"].to_list()
            q70 = data["q70"].to_list()
            q95 = data["q95"].to_list()
            n = data["n"].to_list()

            color = model_colors.get(model, "#00CC96")

            # Plotly allows custom quartiles / whiskers in Box
            fig.add_trace(
                go.Box(
                    name=model,
                    x=x,
                    q1=q20,
                    median=median,
                    q3=q70,
                    lowerfence=q05,
                    upperfence=q95,
                    marker_color=color,
                    legendgroup=model,
                    # Only show model once in legend
                    showlegend=(component == "error_x"),
                    customdata=n,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"Model: {model}<br>"
                        "5%: %{lowerfence:.2f}<br>"
                        "20%: %{q1:.2f}<br>"
                        "Median: %{median:.2f}<br>"
                        "70%: %{q3:.2f}<br>"
                        "95%: %{upperfence:.2f}<br>"
                        "n: %{customdata}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    fig.update_layout(
        title=(f"Marker position errors - " f"{coordinate_system.capitalize()} coordinate system"),
        boxmode="group",
        template="plotly_white",
        height=900,
        width=1500,
        hovermode="closest",
        legend_title_text="Model",
    )

    fig.update_yaxes(
        title_text="Error [mm]",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Error [mm]",
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Error [mm]",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Error norm [mm]",
        row=2,
        col=2,
    )

    fig.update_xaxes(
        title_text="Point",
        tickangle=45,
    )

    fig.show()

    return fig, quantiles


if __name__ == "__main__":
    path_to_data = Path(r".\results\linear_kinematics\ml_keypoints_position_errors.parquet")

    df = pl.read_parquet(path_to_data)
    # list unique tasks
    all_tasks = df.select("task").unique().sort("task").to_series().to_list()
    # Example usage
    subjects_to_plot = ["S10", "S12"]
    tasks_to_plot = all_tasks
    points_to_plot = ["HM2", "HM5", "RS", "US"]
    models_to_plot = ["SynthPose", "RTMPose"]
    # check if all the condition are in the dataframe
    if not all(df.filter(pl.col("subject").is_in(subjects_to_plot)).shape[0] > 0 for subject in subjects_to_plot):
        raise ValueError("Some subjects are not present in the dataframe.")
    if not all(df.filter(pl.col("task").is_in(tasks_to_plot)).shape[0] > 0 for task in tasks_to_plot):
        raise ValueError("Some tasks are not present in the dataframe.")
    if not all(df.filter(pl.col("point_short").is_in(points_to_plot)).shape[0] > 0 for point in points_to_plot):
        raise ValueError("Some points are not present in the dataframe.")
    if not all(df.filter(pl.col("model").is_in(models_to_plot)).shape[0] > 0 for model in models_to_plot):
        raise ValueError("Some models are not present in the dataframe.")

    plot_error_quantiles(
        df=df,
        subjects=subjects_to_plot,
        tasks=tasks_to_plot,
        points=points_to_plot,
        models=models_to_plot,
        coordinate_system="anatomical",
    )
