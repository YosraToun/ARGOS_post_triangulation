from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_robeson = Path(r"./data/pre_processed/Robeson_3Dtrajectories_errors_per_trial.parquet")
path_willmott = Path(r"./data/pre_processed/Willmott_3Dtrajectories_errors_per_trial.parquet")

df_robeson = pl.read_parquet(path_robeson)
df_willmott = pl.read_parquet(path_willmott)

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
# SHARED FILTER
# ============================================================

def filter_data(
    df: pl.DataFrame,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
) -> pl.DataFrame:
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    return df_plot


# ============================================================
# PLOT FUNCTION: ROBESON QUANTILE BOXPLOTS (X, Y, Z, + 3D NORM)
# ============================================================

def plot_robeson_quantiles(
    df: pl.DataFrame,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = filter_data(df, tasks, points, models, coordinate_system)

    # Calculate total MAE per axis
    df_plot = df_plot.with_columns([
        (pl.col("MAE_b_x") + pl.col("MAE_p_x") + pl.col("MAE_u_x")).alias("MAE_total_x"),
        (pl.col("MAE_b_y") + pl.col("MAE_p_y") + pl.col("MAE_u_y")).alias("MAE_total_y"),
        (pl.col("MAE_b_z") + pl.col("MAE_p_z") + pl.col("MAE_u_z")).alias("MAE_total_z"),
    ])

    # 1. Per-axis unpivoting & quantiles
    robeson_columns = [
        "MAE_b_x", "MAE_p_x", "MAE_u_x",
        "MAE_b_y", "MAE_p_y", "MAE_u_y",
        "MAE_b_z", "MAE_p_z", "MAE_u_z",
        "MAE_total_x", "MAE_total_y", "MAE_total_z",
    ]

    df_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "task"],
            on=robeson_columns,
            variable_name="metric",
            value_name="error",
        )
        .drop_nulls("error")
        .with_columns([
            pl.when(pl.col("metric").str.starts_with("MAE_b_")).then(pl.lit("b"))
              .when(pl.col("metric").str.starts_with("MAE_p_")).then(pl.lit("p"))
              .when(pl.col("metric").str.starts_with("MAE_u_")).then(pl.lit("u"))
              .when(pl.col("metric").str.starts_with("MAE_total_")).then(pl.lit("total"))
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
    )

    # 2. 3D Norm unpivoting & quantiles
    df_norm = df_plot.with_columns([
        (pl.col("MAE_b_x").pow(2) + pl.col("MAE_b_y").pow(2) + pl.col("MAE_b_z").pow(2)).sqrt().alias("MAE_b_norm"),
        (pl.col("MAE_p_x").pow(2) + pl.col("MAE_p_y").pow(2) + pl.col("MAE_p_z").pow(2)).sqrt().alias("MAE_p_norm"),
        (pl.col("MAE_u_x").pow(2) + pl.col("MAE_u_y").pow(2) + pl.col("MAE_u_z").pow(2)).sqrt().alias("MAE_u_norm"),
        (pl.col("MAE_total_x").pow(2) + pl.col("MAE_total_y").pow(2) + pl.col("MAE_total_z").pow(2)).sqrt().alias("MAE_total_norm"),
    ])

    norm_columns = ["MAE_b_norm", "MAE_p_norm", "MAE_u_norm", "MAE_total_norm"]
    df_norm_unpivoted = (
        df_norm.unpivot(
            index=["point_short", "model", "task"],
            on=norm_columns,
            variable_name="metric",
            value_name="error",
        )
        .drop_nulls("error")
        .with_columns([
            pl.when(pl.col("metric") == "MAE_b_norm").then(pl.lit("b"))
              .when(pl.col("metric") == "MAE_p_norm").then(pl.lit("p"))
              .when(pl.col("metric") == "MAE_u_norm").then(pl.lit("u"))
              .when(pl.col("metric") == "MAE_total_norm").then(pl.lit("total"))
              .otherwise(pl.lit("unknown"))
              .alias("component"),
            pl.lit("norm").alias("axis"),
        ])
    )

    quantiles_norm = (
        df_norm_unpivoted
        .group_by(["point_short", "model", "axis", "component"])
        .agg(
            pl.col("error").quantile(0.05).alias("q05"),
            pl.col("error").quantile(0.20).alias("q20"),
            pl.col("error").quantile(0.50).alias("median"),
            pl.col("error").quantile(0.70).alias("q70"),
            pl.col("error").quantile(0.95).alias("q95"),
            pl.len().alias("n"),
        )
    )

    quantiles_all = pl.concat([quantiles, quantiles_norm])

    titles = ["X Axis", "Y Axis", "Z Axis", "3D Norm"]
    fig = make_subplots(rows=4, cols=1, subplot_titles=titles, vertical_spacing=0.06)

    positions = {"x": 1, "y": 2, "z": 3, "norm": 4}

    component_colors = {
        "b": "#d62728",     # Systematic Bias (Red)
        "p": "#ff7f0e",     # Proportional Error (Orange)
        "u": "#1f77b4",     # Unsystematic Error (Blue)
        "total": "#2ca02c", # Total MAE (Green)
    }

    component_labels = {
        "b": "Systematic Bias",
        "p": "Proportional Error",
        "u": "Unsystematic Error",
        "total": "Total MAE",
    }

    model_dash = {"SynthPose": "solid", "RTMPose": "dash"}

    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=component_colors[key]),
                name=label,
                legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "b" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, dash_style in model_dash.items():
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

    available_models = quantiles_all.select("model").unique().to_series().to_list()
    available_points = [p for p in POINT_ORDER if p in quantiles_all.get_column("point_short").unique().to_list()]

    for axis_name, row_idx in positions.items():
        axis_data = quantiles_all.filter(pl.col("axis") == axis_name)

        for model_index, model in enumerate(available_models):
            model_data = axis_data.filter(pl.col("model") == model)
            dash_style = model_dash.get(model, "solid")
            components = ["b", "p", "u", "total"]

            for comp_index, component in enumerate(components):
                comp_data = model_data.filter(pl.col("component") == component)
                color = component_colors[component]
                offset = (model_index * 4 + comp_index - 3.5) * 0.05

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

    for row_idx in (1, 2, 3, 4):
        fig.update_xaxes(
            title_text="Anatomical Landmarks" if row_idx == 4 else "",
            tickmode="array",
            tickvals=tickvals,
            ticktext=available_points,
            tickangle=45,
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text="Error (mm)" if row_idx != 4 else " 3D Norm (mm)", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text="<b>Robeson Error Decomposition — Pooled Subjects Per Trial</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        ),
        height=1750,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=220, t=120, b=100),
    )

    return fig


# ============================================================
# PLOT FUNCTION: WILLMOTT QUANTILE BOXPLOTS (X, Y, Z, + 3D NORM)
# ============================================================

def plot_willmott_quantiles(
    df: pl.DataFrame,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = filter_data(df, tasks, points, models, coordinate_system)

    cols = df_plot.columns
    willmott_columns = [
        "RMSE_x", "RMSE_s_x", "RMSE_u_x",
        "RMSE_y", "RMSE_s_y", "RMSE_u_y",
        "RMSE_z", "RMSE_s_z", "RMSE_u_z",
    ]
    willmott_columns = [c for c in willmott_columns if c in cols]

    # 1. Per-axis unpivoting & quantiles
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
    )

    # 2. 3D Norm unpivoting & quantiles
    if "RMSE_x" in cols and "RMSE_y" in cols and "RMSE_z" in cols:
        df_plot = df_plot.with_columns([
            (pl.col("RMSE_x").pow(2) + pl.col("RMSE_y").pow(2) + pl.col("RMSE_z").pow(2)).sqrt().alias("RMSE_norm")
        ])
    if "RMSE_s_x" in cols and "RMSE_s_y" in cols and "RMSE_s_z" in cols:
        df_plot = df_plot.with_columns([
            (pl.col("RMSE_s_x").pow(2) + pl.col("RMSE_s_y").pow(2) + pl.col("RMSE_s_z").pow(2)).sqrt().alias("RMSE_s_norm")
        ])
    if "RMSE_u_x" in cols and "RMSE_u_y" in cols and "RMSE_u_z" in cols:
        df_plot = df_plot.with_columns([
            (pl.col("RMSE_u_x").pow(2) + pl.col("RMSE_u_y").pow(2) + pl.col("RMSE_u_z").pow(2)).sqrt().alias("RMSE_u_norm")
        ])

    norm_columns = [c for c in ["RMSE_norm", "RMSE_s_norm", "RMSE_u_norm"] if c in df_plot.columns]
    df_norm_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "task"],
            on=norm_columns,
            variable_name="metric",
            value_name="error",
        )
        .drop_nulls("error")
        .with_columns([
            pl.when(pl.col("metric") == "RMSE_norm").then(pl.lit("rmse"))
              .when(pl.col("metric") == "RMSE_s_norm").then(pl.lit("s"))
              .when(pl.col("metric") == "RMSE_u_norm").then(pl.lit("u"))
              .otherwise(pl.lit("unknown"))
              .alias("component"),
            pl.lit("norm").alias("axis"),
        ])
    )

    quantiles_norm = (
        df_norm_unpivoted
        .group_by(["point_short", "model", "axis", "component"])
        .agg(
            pl.col("error").quantile(0.05).alias("q05"),
            pl.col("error").quantile(0.20).alias("q20"),
            pl.col("error").quantile(0.50).alias("median"),
            pl.col("error").quantile(0.70).alias("q70"),
            pl.col("error").quantile(0.95).alias("q95"),
            pl.len().alias("n"),
        )
    )

    quantiles_all = pl.concat([quantiles, quantiles_norm])

    titles = ["X Axis", "Y Axis", "Z Axis", "3D Norm"]
    fig = make_subplots(rows=4, cols=1, subplot_titles=titles, vertical_spacing=0.06)

    positions = {"x": 1, "y": 2, "z": 3, "norm": 4}

    component_colors = {
        "rmse": "#2ca02c",  # Total RMSE (Green)
        "s": "#d62728",     # Systematic RMSE (Red)
        "u": "#1f77b4",     # Unsystematic RMSE (Blue)
    }

    component_labels = {
        "rmse": "Total RMSE",
        "s": "Systematic RMSE",
        "u": "Unsystematic RMSE",
    }

    model_dash = {"SynthPose": "solid", "RTMPose": "dash"}

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

    for model_name, dash_style in model_dash.items():
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

    available_models = quantiles_all.select("model").unique().to_series().to_list()
    available_points = [p for p in POINT_ORDER if p in quantiles_all.get_column("point_short").unique().to_list()]

    for axis_name, row_idx in positions.items():
        axis_data = quantiles_all.filter(pl.col("axis") == axis_name)

        for model_index, model in enumerate(available_models):
            model_data = axis_data.filter(pl.col("model") == model)
            dash_style = model_dash.get(model, "solid")
            components = ["rmse", "s", "u"]

            for comp_index, component in enumerate(components):
                comp_data = model_data.filter(pl.col("component") == component)
                color = component_colors[component]
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

    for row_idx in (1, 2, 3, 4):
        fig.update_xaxes(
            title_text="Anatomical Landmarks" if row_idx == 4 else "",
            tickmode="array",
            tickvals=tickvals,
            ticktext=available_points,
            tickangle=45,
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text="Error (mm)" if row_idx != 4 else " 3D Norm (mm)", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text="<b>Willmott Error Decomposition — Pooled Subjects Per Trial </b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18)
        ),
        height=1750,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=220, t=120, b=100),
    )

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASETS
# ============================================================

all_tasks = sorted(list(set(df_robeson.select("task").unique().to_series().to_list() + 
                            df_willmott.select("task").unique().to_series().to_list())))

all_models = sorted(list(set(df_robeson.select("model").unique().to_series().to_list() + 
                             df_willmott.select("model").unique().to_series().to_list())))

points_robeson = set(df_robeson.get_column("point_short").unique().to_list())
points_willmott = set(df_willmott.get_column("point_short").unique().to_list())
common_points = points_robeson.intersection(points_willmott)
all_points = [point for point in POINT_ORDER if point in common_points]


# ============================================================
# DASH APPLICATION LAYOUT
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "3D Trajectories Error Decomposition — Pooled Subjects Per Trial",
            style={"textAlign": "center"},
        ),
        html.H3("Decomposition Method"),
        dcc.Dropdown(
            id="decomposition-selector",
            options=[
                {"label": "Robeson Error Decomposition", "value": "robeson"},
                {"label": "Willmott Error Decomposition", "value": "willmott"},
            ],
            value="robeson",
            clearable=False,
        ),
        html.Hr(),
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
                style={"height": "1800px"},
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
    Output("error-graph", "style"),
    Input("decomposition-selector", "value"),
    Input("task-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_decomposition,
    selected_tasks,
    selected_points,
    selected_models,
):
    if selected_decomposition == "robeson":
        fig = plot_robeson_quantiles(
            df=df_robeson,
            tasks=selected_tasks,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )
    else:
        fig = plot_willmott_quantiles(
            df=df_willmott,
            tasks=selected_tasks,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )

    return fig, {"height": f"{fig.layout.height + 50}px"}


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)