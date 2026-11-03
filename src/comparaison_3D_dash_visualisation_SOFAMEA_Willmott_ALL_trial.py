from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_to_data = Path(r".\data\pre_processed\Willmott_3Dtrajectories_errors_ALL_trial.parquet")

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


# ============================================================
# PLOT FUNCTION
# ============================================================


def plot_error_quantiles(
    df: pl.DataFrame,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    cols = df_plot.columns

    willmott_columns = [
        "RMSE_x", "RMSE_s_x", "RMSE_u_x",
        "RMSE_y", "RMSE_s_y", "RMSE_u_y",
        "RMSE_z", "RMSE_s_z", "RMSE_u_z",
    ]
    willmott_columns = [c for c in willmott_columns if c in cols]

    # --------------------------------------------------------
    # Unpivot Willmott RMSE Decomposition Columns
    # --------------------------------------------------------

    df_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "task"],
            on=willmott_columns,
            variable_name="metric",
            value_name="error",
        )
        .drop_nulls("error")
        .with_columns([
            pl.when(pl.col("metric").str.starts_with("RMSE_s_")).then(pl.lit("s"))
              .when(pl.col("metric").str.starts_with("RMSE_u_")).then(pl.lit("u"))
              .when(pl.col("metric").is_in(["RMSE_x", "RMSE_y", "RMSE_z"])).then(pl.lit("rmse"))
              .otherwise(pl.lit("unknown"))
              .alias("component"),
            pl.when(pl.col("metric").str.ends_with("_x")).then(pl.lit("x"))
              .when(pl.col("metric").str.ends_with("_y")).then(pl.lit("y"))
              .when(pl.col("metric").str.ends_with("_z")).then(pl.lit("z"))
              .otherwise(pl.lit("unknown"))
              .alias("axis"),
        ])
    )

    quantiles = (
        df_unpivoted
        .group_by(["point_short", "model", "axis", "component"])
        .agg(
            pl.col("error").quantile(0.05).alias("q05"),
            pl.col("error").quantile(0.20).alias("q20"),
            pl.col("error").quantile(0.50).alias("median"),
            pl.col("error").quantile(0.70).alias("q70"),
            pl.col("error").quantile(0.95).alias("q95"),
            pl.len().alias("n"),
        )
        .sort(["point_short", "model", "axis", "component"])
    )

    # --------------------------------------------------------
    # Figure Setup (3 Rows, 1 Column)
    # --------------------------------------------------------

    titles = [
        "X Axis ",
        "Y Axis ",
        "Z Axis ",
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.1,
    )

    positions = {
        "x": 1,
        "y": 2,
        "z": 3,
    }

    # Color mappings for RMSE, RMSE_s, RMSE_u
    component_colors = {
        "rmse": "#2ca02c",  # Total RMSE (Green)
        "s": "#d62728",     # Systematic RMSE_s (Red)
        "u": "#1f77b4",     # Unsystematic RMSE_u (Blue)
    }

    component_labels = {
        "rmse": "Total RMSE",
        "s": "Systematic RMSE ",
        "u": "Unsystematic RMSE ",
    }

    # Model Line Style mapping
    model_dash = {
        "SynthPose": "solid",
        "RTMPose": "dash",
    }

    # --------------------------------------------------------
    # Add Grouped & Cleanly Separated Legend Traces
    # --------------------------------------------------------
    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=component_colors[key]),
                name=label,
                legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "rmse" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    model_labels = {
        "SynthPose": "solid",
        "RTMPose": "dash",
    }
    for model_name, dash_style in model_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="gray", width=2.5, dash=dash_style),
                name=model_name,
                legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = quantiles.select("model").unique().to_series().to_list()
    available_points = [p for p in POINT_ORDER if p in quantiles.get_column("point_short").unique().to_list()]

    # --------------------------------------------------------
    # Draw custom percentile boxes per axis
    # --------------------------------------------------------

    for axis_name, row_idx in positions.items():
        axis_data = quantiles.filter(pl.col("axis") == axis_name)

        for model_index, model in enumerate(available_models):
            model_data = axis_data.filter(pl.col("model") == model)
            dash_style = model_dash.get(model, "solid")

            components = ["rmse", "s", "u"]

            for comp_index, component in enumerate(components):
                comp_data = model_data.filter(pl.col("component") == component)
                color = component_colors[component]
                
                # Offset calculation for 2 models x 3 components = 6 items per point
                offset = (model_index * 3 + comp_index - 2.5) * 0.06

                for point_index, point in enumerate(available_points):
                    q = comp_data.filter(pl.col("point_short") == point)
                    if q.height == 0:
                        continue

                    q05, q20, median, q70, q95 = q["q05"][0], q["q20"][0], q["median"][0], q["q70"][0], q["q95"][0]
                    x = point_index + offset

                    fig.add_trace(go.Scatter(x=[x, x], y=[q05, q95], mode="lines", line=dict(color=color, width=1.5, dash=dash_style), showlegend=False, hoverinfo="skip"), row=row_idx, col=1)
                    fig.add_shape(type="rect", x0=x - 0.022, x1=x + 0.022, y0=q20, y1=q70, line=dict(color=color, width=1.5, dash=dash_style), fillcolor=color, opacity=0.3, row=row_idx, col=1)
                    fig.add_trace(go.Scatter(x=[x - 0.022, x + 0.022], y=[median, median], mode="lines", line=dict(color=color, width=2, dash=dash_style), showlegend=False, hoverinfo="skip"), row=row_idx, col=1)

    tickvals = list(range(len(available_points)))

    for row_idx in (1, 2, 3):
        fig.update_xaxes(
            title_text="Anatomical Landmarks",
            tickmode="array",
            tickvals=tickvals,
            ticktext=available_points,
            tickangle=45,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            title_text="RMSE (mm)",
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        height=1300,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=14),
            tracegroupgap=25
        ),
        margin=dict(l=60, r=220, t=100, b=100),
    )

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASET
# ============================================================

all_tasks = df.select("task").unique().sort("task").to_series().to_list()
all_models = df.select("model").unique().sort("model").to_series().to_list()

points_in_data = set(df.get_column("point_short").unique().to_list())
all_points = [point for point in POINT_ORDER if point in points_in_data]


# ============================================================
# DASH APPLICATION LAYOUT
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Willmott Task-Pooled Error Decomposition — All Subjects",
            style={"textAlign": "center"},
        ),
        html.H3("Tasks"),
        dcc.Checklist(
            id="task-selector",
            options=[{"label": task, "value": task} for task in all_tasks],
            value=all_tasks,
            inline=True,
        ),
        html.Hr(),
        html.H3("Points"),
        dcc.Dropdown(
            id="point-selector",
            options=[{"label": point, "value": point} for point in all_points],
            value=all_points,
            multi=True,
        ),
        html.Hr(),
        html.H3("Models"),
        dcc.Checklist(
            id="model-selector",
            options=[{"label": model, "value": model} for model in all_models],
            value=all_models,
            inline=True,
        ),
        html.Hr(),
        dcc.Loading(
            type="circle",
            children=dcc.Graph(
                id="error-graph",
                style={"height": "1350px"},
            ),
        ),
    ],
    style={"width": "95%", "margin": "auto", "fontFamily": "Arial"},
)


# ============================================================
# CALLBACK
# ============================================================


@app.callback(
    Output("error-graph", "figure"),
    Input("task-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_tasks,
    selected_points,
    selected_models,
):
    return plot_error_quantiles(
        df=df,
        tasks=selected_tasks,
        points=selected_points,
        models=selected_models,
        coordinate_system="anatomical",
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)