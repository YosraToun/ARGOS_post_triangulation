from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_robeson = Path(r"./data/pre_processed/Robeson_3Dtrajectories_errors_ALL.parquet")
path_willmott = Path(r"./data/pre_processed/Willmott_3Dtrajectories_errors_ALL.parquet")

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
# PLOT FUNCTION: ROBESON
# ============================================================

def plot_robeson_barplots(
    df: pl.DataFrame,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    df_plot = df_plot.with_columns([
        (pl.col("MAE_b_x") + pl.col("MAE_p_x") + pl.col("MAE_u_x")).alias("MAE_total_x"),
        (pl.col("MAE_b_y") + pl.col("MAE_p_y") + pl.col("MAE_u_y")).alias("MAE_total_y"),
        (pl.col("MAE_b_z") + pl.col("MAE_p_z") + pl.col("MAE_u_z")).alias("MAE_total_z"),
    ])

    agg_cols = [c for c in df_plot.columns if c.startswith("MAE_")]
    df_plot = (
        df_plot
        .group_by(["point_short", "model"])
        .agg([pl.col(c).mean() for c in agg_cols])
    )

    mapping_df = pl.DataFrame({
        "point_short": POINT_ORDER,
        "sort_idx": range(len(POINT_ORDER))
    })

    df_plot = (
        df_plot
        .join(mapping_df, on="point_short", how="inner")
        .sort(["sort_idx", "model"])
    )

    titles = ["X Axis", "Y Axis", "Z Axis"]
    fig = make_subplots(rows=3, cols=1, subplot_titles=titles, vertical_spacing=0.12)

    axes_config = [("x", 1, 1), ("y", 2, 1), ("z", 3, 1)]

    component_colors = {
        "b": "#d62728",     # Systematic Bias (Red)
        "p": "#ff7f0e",     # Proportional Error (Orange)
        "u": "#1f77b4",     # Unsystematic Error (Blue)
        "total": "#2ca02c", # Total MAE (Green)
    }

    component_labels = {
        "total": "Total MAE",
        "b": "Systematic Bias",
        "p": "Proportional Error",
        "u": "Unsystematic Error",
    }

    model_patterns = {"SynthPose": "", "RTMPose": "/"}

    current_points = df_plot.get_column("point_short").unique().to_list()
    ordered_points = [p for p in POINT_ORDER if p in current_points]
    anchor_x = ordered_points[0] if ordered_points else "C7"

    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=18, color=component_colors[key], symbol="square"),
                name=label,
                legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "total" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, pattern in model_patterns.items():
        legend_name = f"{model_name} (Hatched)" if model_name == "RTMPose" else model_name
        fig.add_trace(
            go.Bar(
                x=[anchor_x], y=[None],
                offsetgroup=model_name,
                marker=dict(color="gray", pattern_shape=pattern, line=dict(color="black", width=0.5)),
                name=legend_name,
                legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = df_plot.get_column("model").unique().to_list()

    for axis_name, row_idx, col_idx in axes_config:
        for model in available_models:
            model_df = df_plot.filter(pl.col("model") == model)
            if model_df.height == 0:
                continue

            x_labels = model_df.get_column("point_short").to_list()
            pattern = model_patterns.get(model, "")

            for comp_key in ["total", "b", "p", "u"]:
                col_name = f"MAE_total_{axis_name}" if comp_key == "total" else f"MAE_{comp_key}_{axis_name}"
                if col_name not in model_df.columns:
                    continue

                y_data = model_df.get_column(col_name).to_list()

                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_data,
                        name=component_labels[comp_key],
                        offsetgroup=f"{model}_{comp_key}",
                        marker=dict(color=component_colors[comp_key], pattern_shape=pattern, line=dict(width=0)),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>%{{x}}</b> ({model})<br>"
                            + f"{component_labels[comp_key]}: "
                            + "%{y:.2f} mm<br>"
                            + "<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=col_idx,
                )

    fig.update_layout(
        barmode="group",
        bargroupgap=0.15,
        height=1200,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=240, t=100, b=100),
    )

    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ordered_points)

    for axis_name, row_idx, col_idx in axes_config:
        fig.update_yaxes(title_text="MAE (mm)", row=row_idx, col=col_idx)
        fig.update_xaxes(title_text="Anatomical Landmarks", tickangle=45, row=row_idx, col=col_idx)

    return fig


# ============================================================
# PLOT FUNCTION: WILLMOTT
# ============================================================

def plot_willmott_barplots(
    df: pl.DataFrame,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    agg_cols = [c for c in df_plot.columns if c.startswith("RMSE_")]
    df_plot = (
        df_plot
        .group_by(["point_short", "model"])
        .agg([pl.col(c).mean() for c in agg_cols])
    )

    mapping_df = pl.DataFrame({
        "point_short": POINT_ORDER,
        "sort_idx": range(len(POINT_ORDER))
    })

    df_plot = (
        df_plot
        .join(mapping_df, on="point_short", how="inner")
        .sort(["sort_idx", "model"])
    )

    titles = ["X Axis", "Y Axis", "Z Axis"]
    fig = make_subplots(rows=3, cols=1, subplot_titles=titles, vertical_spacing=0.12)

    axes_config = [("x", 1, 1), ("y", 2, 1), ("z", 3, 1)]

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

    model_patterns = {"SynthPose": "", "RTMPose": "/"}

    current_points = df_plot.get_column("point_short").unique().to_list()
    ordered_points = [p for p in POINT_ORDER if p in current_points]
    anchor_x = ordered_points[0] if ordered_points else "C7"

    for key, label in component_labels.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=18, color=component_colors[key], symbol="square"),
                name=label,
                legendgroup="components",
                legendgrouptitle_text="<b>Error Components</b>" if key == "rmse" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, pattern in model_patterns.items():
        legend_name = f"{model_name} (Hatched)" if model_name == "RTMPose" else model_name
        fig.add_trace(
            go.Bar(
                x=[anchor_x], y=[None],
                offsetgroup=model_name,
                marker=dict(color="gray", pattern_shape=pattern, line=dict(color="black", width=0.5)),
                name=legend_name,
                legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = df_plot.get_column("model").unique().to_list()

    for axis_name, row_idx, col_idx in axes_config:
        for model in available_models:
            model_df = df_plot.filter(pl.col("model") == model)
            if model_df.height == 0:
                continue

            x_labels = model_df.get_column("point_short").to_list()
            pattern = model_patterns.get(model, "")

            for comp_key in ["rmse", "s", "u"]:
                col_name = f"RMSE_{axis_name}" if comp_key == "rmse" else f"RMSE_{comp_key}_{axis_name}"
                if col_name not in model_df.columns:
                    continue

                y_data = model_df.get_column(col_name).to_list()

                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_data,
                        name=component_labels[comp_key],
                        offsetgroup=f"{model}_{comp_key}",
                        marker=dict(color=component_colors[comp_key], pattern_shape=pattern, line=dict(width=0)),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>%{{x}}</b> ({model})<br>"
                            + f"{component_labels[comp_key]}: "
                            + "%{y:.2f} mm<br>"
                            + "<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=col_idx,
                )

    fig.update_layout(
        barmode="group",
        bargroupgap=0.15,
        height=1200,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=14), tracegroupgap=25),
        margin=dict(l=60, r=240, t=100, b=100),
    )

    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ordered_points)

    for axis_name, row_idx, col_idx in axes_config:
        fig.update_yaxes(title_text="RMSE (mm)", row=row_idx, col=col_idx)
        fig.update_xaxes(title_text="Anatomical Landmarks", tickangle=45, row=row_idx, col=col_idx)

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASETS
# ============================================================

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
            "3D Trajectories Error Decomposition — All Subjects & Trials",
            style={"textAlign": "center"},
        ),
        html.H3("Decomposition Method"),
        dcc.Dropdown(
            id="decomposition-selector",
            options=[
                {"label": "Robeson (MAE Decomposition: Bias, Proportional, Unsystematic)", "value": "robeson"},
                {"label": "Willmott (RMSE Decomposition: Systematic, Unsystematic)", "value": "willmott"},
            ],
            value="robeson",
            clearable=False,
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
                style={"height": "1250px"},
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
    Input("decomposition-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_decomposition,
    selected_points,
    selected_models,
):
    if selected_decomposition == "robeson":
        return plot_robeson_barplots(
            df=df_robeson,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )
    else:
        return plot_willmott_barplots(
            df=df_willmott,
            points=selected_points,
            models=selected_models,
            coordinate_system="anatomical",
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
