from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_to_data = Path(r".\results\linear_kinematics\ml_keypoints_position_errors.parquet")
df = pl.read_parquet(path_to_data)

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


# ============================================================
# PLOT FUNCTION
# ============================================================


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
    """

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    components = [
        "error_x",
        "error_y",
        "error_z",
        "error_norm",
    ]

    titles = [
        "X error",
        "Y error",
        "Z error",
        "Error norm",
    ]

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

    model_colors = {
        "SynthPose": "#1f77b4",
        "RTMPose": "#ff7f0e",
        "SynthRTMPose": "#00CC96",
    }

    models_in_data = set(quantiles.get_column("model").unique().to_list())
    available_models = [m for m in MODEL_ORDER if m in models_in_data] + sorted(
        m for m in models_in_data if m not in MODEL_ORDER
    )

    points_in_data = set(quantiles.get_column("point_short").unique().to_list())
    available_points = [p for p in POINT_ORDER if p in points_in_data] + sorted(
        p for p in points_in_data if p not in POINT_ORDER
    )

    n_models = max(len(available_models), 1)
    # Slightly narrower offset when 3 models are displayed side by side
    step_width = 0.22 if n_models >= 3 else 0.25
    box_half_width = 0.075 if n_models >= 3 else 0.09

    # --------------------------------------------------------
    # Draw custom percentile boxes
    # --------------------------------------------------------

    for component in components:

        row, col = positions[component]

        q_component = quantiles.filter(pl.col("component") == component)

        for model_index, model in enumerate(available_models):

            q_model = q_component.filter(pl.col("model") == model)

            color = model_colors.get(model, "#636EFA")

            # Offset models slightly so they do not overlap
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
                n = q["n"][0]

                x = point_index + offset

                # --------------------------------------------
                # 5 -> 95 percentile whisker
                # --------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[q05, q95],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=2,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Box 20 -> 70 percentile
                # --------------------------------------------

                fig.add_shape(
                    type="rect",
                    x0=x - box_half_width,
                    x1=x + box_half_width,
                    y0=q20,
                    y1=q70,
                    line=dict(
                        color=color,
                        width=2,
                    ),
                    fillcolor=color,
                    opacity=0.25,
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Median
                # --------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=[x - box_half_width, x + box_half_width],
                        y=[median, median],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=3,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Hover point
                # --------------------------------------------

                show_leg = (component == "error_x") and (not legend_added)
                if show_leg:
                    legend_added = True

                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[median],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=8,
                        ),
                        name=model,
                        legendgroup=model,
                        showlegend=show_leg,
                        customdata=[
                            [
                                point,
                                model,
                                q05,
                                q20,
                                median,
                                q70,
                                q95,
                                n,
                            ]
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Model: %{customdata[1]}<br>"
                            "5%: %{customdata[2]:.2f}<br>"
                            "20%: %{customdata[3]:.2f}<br>"
                            "Median: %{customdata[4]:.2f}<br>"
                            "70%: %{customdata[5]:.2f}<br>"
                            "95%: %{customdata[6]:.2f}<br>"
                            "n: %{customdata[7]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

    # --------------------------------------------------------
    # X and Y axis labels
    # --------------------------------------------------------

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
        template="plotly_white",
        title=dict(
            text=f"<b>Markerless position errors - {coordinate_system.capitalize()} coordinate system</b>",
            x=0.5,
            xanchor="center",
        ),
        hovermode="closest",
        margin=dict(
            l=60,
            r=30,
            t=80,
            b=100,
        ),
    )

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASET
# ============================================================

all_tasks = df.select("task").unique().sort("task").to_series().to_list()
all_subjects = df.select("subject").unique().sort("subject").to_series().to_list()

points_in_df = set(df.get_column("point_short").unique().to_list())
all_points = [p for p in POINT_ORDER if p in points_in_df] + sorted(
    p for p in points_in_df if p not in POINT_ORDER
)

models_in_df = set(df.get_column("model").unique().to_list())
all_models = [m for m in MODEL_ORDER if m in models_in_df] + sorted(
    m for m in models_in_df if m not in MODEL_ORDER
)

all_coord_systems = df.select("coordinate_system").unique().sort("coordinate_system").to_series().to_list()
default_coord = "anatomical" if "anatomical" in all_coord_systems else all_coord_systems[0]


# ============================================================
# DASH APPLICATION
# ============================================================

app = Dash(__name__)

# ============================================================
# DASH LAYOUT
# ============================================================

app.layout = html.Div(
    [
        html.H1(
            "Markerless Position Error Analysis",
            style={"textAlign": "center"},
        ),
        # ----------------------------------------------------
        # COORDINATE SYSTEM
        # ----------------------------------------------------
        html.H3("Coordinate system"),
        dcc.RadioItems(
            id="coord-selector",
            options=[{"label": c.capitalize(), "value": c} for c in all_coord_systems],
            value=default_coord,
            inline=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # TASKS
        # ----------------------------------------------------
        html.H3("Tasks"),
        dcc.Checklist(
            id="task-selector",
            options=[{"label": task, "value": task} for task in all_tasks],
            value=all_tasks,
            inline=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # SUBJECTS
        # ----------------------------------------------------
        html.H3("Subjects"),
        dcc.Dropdown(
            id="subject-selector",
            options=[{"label": subject, "value": subject} for subject in all_subjects],
            value=all_subjects,
            multi=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------
        html.H3("Points"),
        dcc.Dropdown(
            id="point-selector",
            options=[{"label": point, "value": point} for point in all_points],
            value=all_points,
            multi=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------
        html.H3("Models"),
        dcc.Checklist(
            id="model-selector",
            options=[{"label": model, "value": model} for model in all_models],
            value=all_models,
            inline=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------
        dcc.Loading(
            type="circle",
            children=dcc.Graph(
                id="error-graph",
                style={"height": "900px"},
            ),
        ),
    ],
    style={
        "width": "95%",
        "margin": "auto",
        "fontFamily": "Arial",
    },
)


# ============================================================
# CALLBACK
# ============================================================


@app.callback(
    Output("error-graph", "figure"),
    Input("coord-selector", "value"),
    Input("task-selector", "value"),
    Input("subject-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_coord,
    selected_tasks,
    selected_subjects,
    selected_points,
    selected_models,
):
    return plot_error_quantiles(
        df=df,
        tasks=selected_tasks,
        subjects=selected_subjects,
        points=selected_points,
        models=selected_models,
        coordinate_system=selected_coord,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)