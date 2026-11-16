from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# CONSTANTS & POINT ORDER
# ============================================================

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
# PLOT FUNCTIONS FOR BOTH DECOMPOSITIONS
# ============================================================

def plot_robeson_quantiles(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    # Compute Total MAE per axis & 3D Norms across components
    df_plot = df_plot.with_columns([
        (pl.col("MAE_b_x") + pl.col("MAE_p_x") + pl.col("MAE_u_x")).alias("MAE_total_x"),
        (pl.col("MAE_b_y") + pl.col("MAE_p_y") + pl.col("MAE_u_y")).alias("MAE_total_y"),
        (pl.col("MAE_b_z") + pl.col("MAE_p_z") + pl.col("MAE_u_z")).alias("MAE_total_z"),
    ]).with_columns([
        (pl.col("MAE_b_x")**2 + pl.col("MAE_b_y")**2 + pl.col("MAE_b_z")**2).sqrt().alias("MAE_b_norm_3d"),
        (pl.col("MAE_p_x")**2 + pl.col("MAE_p_y")**2 + pl.col("MAE_p_z")**2).sqrt().alias("MAE_p_norm_3d"),
        (pl.col("MAE_u_x")**2 + pl.col("MAE_u_y")**2 + pl.col("MAE_u_z")**2).sqrt().alias("MAE_u_norm_3d"),
        (pl.col("MAE_total_x")**2 + pl.col("MAE_total_y")**2 + pl.col("MAE_total_z")**2).sqrt().alias("MAE_total_norm_3d"),
    ])

    robeson_columns = [
        "MAE_b_x", "MAE_p_x", "MAE_u_x",
        "MAE_b_y", "MAE_p_y", "MAE_u_y",
        "MAE_b_z", "MAE_p_z", "MAE_u_z",
        "MAE_total_x", "MAE_total_y", "MAE_total_z",
        "MAE_b_norm_3d", "MAE_p_norm_3d", "MAE_u_norm_3d", "MAE_total_norm_3d",
    ]

    df_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "subject"],
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
              .when(pl.col("metric").str.ends_with("_norm_3d")).then(pl.lit("norm_3d"))
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

    titles = ["X Axis", "Y Axis", "Z Axis", "3D Norm"]
    fig = make_subplots(rows=4, cols=1, subplot_titles=titles, vertical_spacing=0.08)

    positions = {"x": 1, "y": 2, "z": 3, "norm_3d": 4}
    component_colors = {
        "b": "#d62728",     # Systematic Bias
        "p": "#ff7f0e",     # Proportional Error
        "u": "#1f77b4",     # Unsystematic Error
        "total": "#2ca02c", # Total MAE
    }
    model_dash = {"SynthPose": "solid", "RTMPose": "dash"}

    component_labels = {
        "b": "Systematic Bias",
        "p": "Proportional Error",
        "u": "Unsystematic Error",
        "total": "Total MAE",
    }
    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=component_colors[key]),
                name=label, legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "b" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, dash_style in model_dash.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="gray", width=2.5, dash=dash_style),
                name=model_name, legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = quantiles.select("model").unique().to_series().to_list()
    available_points = [p for p in POINT_ORDER if p in quantiles.get_column("point_short").unique().to_list()]

    for axis_name, row_idx in positions.items():
        axis_data = quantiles.filter(pl.col("axis") == axis_name)

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
            tickmode="array", tickvals=tickvals, ticktext=available_points,
            tickangle=45, row=row_idx, col=1,
        )
        fig.update_yaxes(title_text="Error (mm)" if row_idx != 4 else " 3D Norm (mm)", row=row_idx, col=1)

    fig.update_layout(
        title=dict(text="<b>Robeson Error Decomposition — Pooled trials Per subject</b>", font=dict(size=22), x=0.5, xanchor="center", yanchor="top"),
        height=1650, template="plotly_white", hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=220, t=100, b=100),
    )
    return fig


def plot_willmott_quantiles(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    df_plot = df_plot.with_columns([
        (pl.col("RMSE_x")**2 + pl.col("RMSE_y")**2 + pl.col("RMSE_z")**2).sqrt().alias("RMSE_norm_3d"),
        (pl.col("RMSE_s_x")**2 + pl.col("RMSE_s_y")**2 + pl.col("RMSE_s_z")**2).sqrt().alias("RMSE_s_norm_3d"),
        (pl.col("RMSE_u_x")**2 + pl.col("RMSE_u_y")**2 + pl.col("RMSE_u_z")**2).sqrt().alias("RMSE_u_norm_3d"),
    ])

    willmott_columns = [
        "RMSE_x", "RMSE_s_x", "RMSE_u_x",
        "RMSE_y", "RMSE_s_y", "RMSE_u_y",
        "RMSE_z", "RMSE_s_z", "RMSE_u_z",
        "RMSE_norm_3d", "RMSE_s_norm_3d", "RMSE_u_norm_3d",
    ]

    df_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "subject"],
            on=willmott_columns,
            variable_name="metric",
            value_name="error",
        )
        .drop_nulls("error")
        .with_columns([
            pl.when(pl.col("metric").str.starts_with("RMSE_s_")).then(pl.lit("s"))
              .when(pl.col("metric").str.starts_with("RMSE_u_")).then(pl.lit("u"))
              .when(pl.col("metric").str.starts_with("RMSE_") & ~pl.col("metric").str.contains("_s_") & ~pl.col("metric").str.contains("_u_")).then(pl.lit("total"))
              .otherwise(pl.lit("unknown"))
              .alias("component"),
            pl.when(pl.col("metric").str.ends_with("_x")).then(pl.lit("x"))
              .when(pl.col("metric").str.ends_with("_y")).then(pl.lit("y"))
              .when(pl.col("metric").str.ends_with("_z")).then(pl.lit("z"))
              .when(pl.col("metric").str.ends_with("_norm_3d")).then(pl.lit("norm_3d"))
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

    titles = ["X Axis", "Y Axis", "Z Axis", "3D Norm"]
    fig = make_subplots(rows=4, cols=1, subplot_titles=titles, vertical_spacing=0.08)

    positions = {"x": 1, "y": 2, "z": 3, "norm_3d": 4}
    component_colors = {
        "s": "#d62728",     # Systematic RMSE
        "u": "#1f77b4",     # Unsystematic RMSE
        "total": "#2ca02c", # Total RMSE
    }
    model_dash = {"SynthPose": "solid", "RTMPose": "dash"}

    component_labels = {
        "s": "Systematic RMSE",
        "u": "Unsystematic RMSE",
        "total": "Total RMSE",
    }
    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=component_colors[key]),
                name=label, legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "s" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, dash_style in model_dash.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="gray", width=2.5, dash=dash_style),
                name=model_name, legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = quantiles.select("model").unique().to_series().to_list()
    available_points = [p for p in POINT_ORDER if p in quantiles.get_column("point_short").unique().to_list()]

    for axis_name, row_idx in positions.items():
        axis_data = quantiles.filter(pl.col("axis") == axis_name)

        for model_index, model in enumerate(available_models):
            model_data = axis_data.filter(pl.col("model") == model)
            dash_style = model_dash.get(model, "solid")
            components = ["s", "u", "total"]

            for comp_index, component in enumerate(components):
                comp_data = model_data.filter(pl.col("component") == component)
                color = component_colors[component]
                offset = (model_index * 3 + comp_index - 2.5) * 0.05

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
            tickmode="array", tickvals=tickvals, ticktext=available_points,
            tickangle=45, row=row_idx, col=1,
        )
        fig.update_yaxes(title_text="Error (mm)" if row_idx != 4 else " 3D Norm (mm)", row=row_idx, col=1)

    fig.update_layout(
        title=dict(text="<b>Willmott Error Decomposition — Pooled trials Per subject</b>", font=dict(size=22), x=0.5, xanchor="center", yanchor="top"),
        height=1650, template="plotly_white", hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=220, t=100, b=100),
    )
    return fig


# ============================================================
# DASH APPLICATION LAYOUT
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "3D Trajectories Error Decomposition Visualization — Pooled trials Per subject",
            style={"textAlign": "center"},
        ),
        html.Hr(),
        html.H3("Decomposition Method"),
        dcc.Dropdown(
            id="decomposition-selector",
            options=[
                {"label": "Robeson Decomposition", "value": "Robeson"},
                {"label": "Willmott Decomposition", "value": "Willmott"},
            ],
            value="Robeson",
            clearable=False,
        ),
        html.Hr(),
        html.H3("Subjects"),
        dcc.Dropdown(
            id="subject-selector",
            multi=True,
        ),
        html.Hr(),
        html.H3("Points"),
        dcc.Dropdown(
            id="point-selector",
            multi=True,
        ),
        html.Hr(),
        html.H3("Models"),
        dcc.Checklist(
            id="model-selector",
            inline=True,
        ),
        html.Hr(),
        dcc.Loading(
            type="circle",
            children=dcc.Graph(
                id="error-graph",
                style={"height": "1700px"},
            ),
        ),
    ],
    style={"width": "95%", "margin": "auto", "fontFamily": "Arial"},
)


# ============================================================
# CALLBACKS TO UPDATE DROPDOWNS DYNAMICALLY BASED ON DATASET
# ============================================================

@app.callback(
    Output("subject-selector", "options"),
    Output("subject-selector", "value"),
    Output("point-selector", "options"),
    Output("point-selector", "value"),
    Output("model-selector", "options"),
    Output("model-selector", "value"),
    Input("decomposition-selector", "value"),
)
def update_dropdowns(decomposition_method):
    if decomposition_method == "Robeson":
        path = Path(r".\data\pre_processed\Robeson_3Dtrajectories_errors_per_subject.parquet")
    else:
        path = Path(r".\data\pre_processed\Willmott_3Dtrajectories_errors_per_subject.parquet")

    df_local = pl.read_parquet(path)

    subjects = df_local.select("subject").unique().sort("subject").to_series().to_list()
    models = df_local.select("model").unique().sort("model").to_series().to_list()
    
    points_in_data = set(df_local.get_column("point_short").unique().to_list())
    points = [point for point in POINT_ORDER if point in points_in_data]

    subject_options = [{"label": s, "value": s} for s in subjects]
    point_options = [{"label": p, "value": p} for p in points]
    model_options = [{"label": m, "value": m} for m in models]

    return (
        subject_options, subjects,
        point_options, points,
        model_options, models
    )


# ============================================================
# MAIN CALLBACK FOR GRAPH GENERATION
# ============================================================

@app.callback(
    Output("error-graph", "figure"),
    Input("decomposition-selector", "value"),
    Input("subject-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    decomposition_method,
    selected_subjects,
    selected_points,
    selected_models,
):
    if not selected_subjects or not selected_points or not selected_models:
        # Return empty figure if selections are empty
        return go.Figure()

    if decomposition_method == "Robeson":
        path = Path(r".\data\pre_processed\Robeson_3Dtrajectories_errors_per_subject.parquet")
        df_local = pl.read_parquet(path)
        return plot_robeson_quantiles(
            df=df_local,
            subjects=selected_subjects,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )
    else:
        path = Path(r".\data\pre_processed\Willmott_3Dtrajectories_errors_per_subject.parquet")
        df_local = pl.read_parquet(path)
        return plot_willmott_quantiles(
            df=df_local,
            subjects=selected_subjects,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)