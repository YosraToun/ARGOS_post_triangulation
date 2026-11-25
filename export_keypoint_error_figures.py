from pathlib import Path
import re

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURATION
# ============================================================

PATH_TO_DATA = Path(r".\results\linear_kinematics\ml_keypoints_position_errors.parquet")
OUTPUT_ROOT = Path(r".\results\linear_kinematics\figures")
COORDINATE_SYSTEM = "anatomical"

DICT_TASK = {
    "00-static-stand": "static",
    "01-eat-yaourt": "eat",
    "02-cut-food": "cut",
    "03-open-bottle": "open bottle",
    "03-Open_a_Bottle_and_Pour": "open and pour",
    "04-Hand_to_Head": "hand to head",
    "04-take-in-box": "take in box",
    "05-use-computer": "computer",
    "06-drawing": "drawing",
    "06-Open_Kit": "open kit",
    "07-use-ruler": "ruler",
    "08-cut-scissor": "cut scissor",
    "09-erase-eraser": "erase eraser",
    "10-reach_and_grasp_far": "R&G far",
    "11-reach_and_grasp_mid": "R&G mid",
    "12-reach_and_grasp_near": "R&G near",
    "13-playdoe": "playdoe",
    "14-put-toothpaste": "put toothpaste",
    "15-brush-teeth": "brush teeth",
    "16-comb-hair": "comb hair",
    "17-hand-to-back": "hand to back",
    "18-lift-tray": "lift tray",
    "20-catch-throw-ball": "catch throw ball",
    "21-abduction": "GH abduction",
    "22-flexion": "GH flexion",
    "23-rotation-plus-supination-full-arm-extended-front": "GH rot + El sup extend",
    "24-Elbow_Flexion_Extension": "Elbow flexion",
    "24-Elbow_Pronosupination_Bras_Bend": "Elbow pronosupination",
    "26-Cymbals": "cymbals",
}

POINT_ORDER = [
    "C7",
    "IJ",
    "T10",
    "LS",
    "GH",
    "EJC",
    "EM",
    "EL",
    "WJC",
    "US",
    "RS",
    "HM2",
    "HM5",
]

MODEL_ORDER = [
    "SynthPose",
    "RTMPose",
    "SynthRTMPose",
]

MODEL_COLORS = {
    "SynthPose": "#1f77b4",
    "RTMPose": "#ff7f0e",
    "SynthRTMPose": "#00CC96",
}


# ============================================================
# HELPERS
# ============================================================


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", name).strip()


def get_task_label(task: str, dict_task: dict[str, str] = DICT_TASK) -> str:
    return dict_task.get(task, task)


# ============================================================
# PLOT FUNCTION
# ============================================================


def build_error_quantiles_figure(
    df: pl.DataFrame,
    title_text: str,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
) -> go.Figure:
    """
    Build figure using:
        whisker low  = 5th percentile
        box low      = 20th percentile
        median       = 50th percentile
        box high     = 70th percentile
        whisker high = 95th percentile
    """
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    error_columns = ["error_x", "error_y", "error_z", "error_norm"]

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

    components = ["error_x", "error_y", "error_z", "error_norm"]
    titles = ["X error", "Y error", "Z error", "Error norm"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
    )

    positions = {
        "error_x": (1, 1),
        "error_y": (1, 2),
        "error_z": (2, 1),
        "error_norm": (2, 2),
    }

    models_in_data = set(quantiles.get_column("model").unique().to_list()) if quantiles.height > 0 else set()
    available_models = [m for m in MODEL_ORDER if m in models_in_data] + sorted(
        m for m in models_in_data if m not in MODEL_ORDER
    )

    points_in_data = set(quantiles.get_column("point_short").unique().to_list()) if quantiles.height > 0 else set()
    available_points = [p for p in POINT_ORDER if p in points_in_data] + sorted(
        p for p in points_in_data if p not in POINT_ORDER
    )

    n_models = max(len(available_models), 1)
    step_width = 0.22 if n_models >= 3 else 0.25
    box_half_width = 0.075 if n_models >= 3 else 0.09

    for component in components:
        row, col = positions[component]
        q_component = quantiles.filter(pl.col("component") == component)

        for model_index, model in enumerate(available_models):
            q_model = q_component.filter(pl.col("model") == model)
            color = MODEL_COLORS.get(model, "#636EFA")
            offset = (model_index - (n_models - 1) / 2) * step_width
            legend_added = False

            for point_index, point in enumerate(available_points):
                q = q_model.filter(pl.col("point_short") == point)
                if q.height == 0:
                    continue

                q05 = q["q05"][0]
                q20 = q["q20"][0]
                median = q["median"][0]
                q70 = q["q70"][0]
                q95 = q["q95"][0]

                x = point_index + offset

                # 5 -> 95 percentile whisker
                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[q05, q95],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # Box 20 -> 70 percentile
                fig.add_shape(
                    type="rect",
                    x0=x - box_half_width,
                    x1=x + box_half_width,
                    y0=q20,
                    y1=q70,
                    line=dict(color=color, width=2),
                    fillcolor=color,
                    opacity=0.25,
                    row=row,
                    col=col,
                )

                # Median
                fig.add_trace(
                    go.Scatter(
                        x=[x - box_half_width, x + box_half_width],
                        y=[median, median],
                        mode="lines",
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # Marker trace for legend
                show_leg = (component == "error_x") and (not legend_added)
                if show_leg:
                    legend_added = True

                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[median],
                        mode="markers",
                        marker=dict(color=color, size=8),
                        name=model,
                        legendgroup=model,
                        showlegend=show_leg,
                    ),
                    row=row,
                    col=col,
                )

    tickvals = list(range(len(available_points)))

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=available_points,
                tickangle=45,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="Error (mm)",
                row=row,
                col=col,
            )

    fig.update_layout(
        height=850,
        width=1500,
        template="plotly_white",
        title=dict(
            text=f"<b>{title_text}</b>",
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=60, r=30, t=80, b=100),
    )

    return fig


# ============================================================
# BATCH EXPORT
# ============================================================


def export_all_figures(
    df: pl.DataFrame,
    output_root: Path = OUTPUT_ROOT,
    coordinate_system: str = COORDINATE_SYSTEM,
    models: list[str] | None = None,
    scale: int = 2,
):
    df_coord = df.filter(pl.col("coordinate_system") == coordinate_system)
    if models is not None:
        df_coord = df_coord.filter(pl.col("model").is_in(models))

    dir_by_subject = output_root / "by_subject" / coordinate_system
    dir_by_task = output_root / "by_task" / coordinate_system
    dir_by_subject.mkdir(parents=True, exist_ok=True)
    dir_by_task.mkdir(parents=True, exist_ok=True)

    subjects = df_coord.select("subject").unique().sort("subject").to_series().to_list()
    tasks = df_coord.select("task").unique().sort("task").to_series().to_list()

    # 1. All trials per subject
    for subject in subjects:
        fig = build_error_quantiles_figure(
            df=df_coord,
            subjects=[subject],
            models=models,
            coordinate_system=coordinate_system,
            title_text=f"Keypoint Position Errors — {subject} (pooled trials)",
        )
        out_path = dir_by_subject / f"{sanitize_filename(subject)}.png"
        fig.write_image(out_path, width=1500, height=850, scale=scale)
        print(f"Saved: {out_path}")

    # 2. All subjects per trial
    for task in tasks:
        task_label = get_task_label(task)
        fig = build_error_quantiles_figure(
            df=df_coord,
            tasks=[task],
            models=models,
            coordinate_system=coordinate_system,
            title_text=f"Keypoint Position Errors — {task_label} (pooled subjects)",
        )
        out_path = dir_by_task / f"{sanitize_filename(task_label)}.png"
        fig.write_image(out_path, width=1500, height=850, scale=scale)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    df_all = pl.read_parquet(PATH_TO_DATA)
    export_all_figures(
        df=df_all,
        output_root=OUTPUT_ROOT,
        coordinate_system=COORDINATE_SYSTEM,
        models=MODEL_ORDER,
        scale=2,
    )