from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_to_data = Path(r".\data\pre_processed\Willmott_3Dtrajectories_errors_ALL.parquet")

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
# PLOT FUNCTION (BAR PLOT PER AXIS)
# ============================================================


def plot_error_barplots(
    df: pl.DataFrame,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    # S'assurer d'avoir des données uniques par point et par modèle (évite l'empilement)
    agg_cols = [c for c in df_plot.columns if c.startswith("RMSE_")]
    df_plot = (
        df_plot
        .group_by(["point_short", "model"])
        .agg([pl.col(c).mean() for c in agg_cols])
    )

    # Sort according to POINT_ORDER
    mapping_df = pl.DataFrame({
        "point_short": POINT_ORDER,
        "sort_idx": range(len(POINT_ORDER))
    })

    df_plot = (
        df_plot
        .join(mapping_df, on="point_short", how="inner")
        .sort(["sort_idx", "model"])
    )

    # --------------------------------------------------------
    # Figure Setup (3 Subplots: X, Y, Z Axes)
    # --------------------------------------------------------

    titles = [
        "X Axis",
        "Y Axis",
        "Z Axis",
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.12,
    )

    axes_config = [
        ("x", 1, 1),
        ("y", 2, 1),
        ("z", 3, 1),
    ]

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

    model_patterns = {
        "SynthPose": "",
        "RTMPose": "/",
    }

    current_points = df_plot.get_column("point_short").unique().to_list()
    ordered_points = [p for p in POINT_ORDER if p in current_points]
    anchor_x = ordered_points[0] if ordered_points else "C7"

    # --------------------------------------------------------
    # Add Legend Traces
    # --------------------------------------------------------
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
                marker=dict(
                    color="gray",
                    pattern_shape=pattern,
                    line=dict(color="black", width=0.5)
                ),
                name=legend_name,
                legendgroup="models",
                legendgrouptitle_text="<b>Models</b>" if model_name == "SynthPose" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    available_models = df_plot.get_column("model").unique().to_list()

    # --------------------------------------------------------
    # Add Grouped Bar Traces for each Axis
    # --------------------------------------------------------
    for axis_name, row_idx, col_idx in axes_config:
        for model in available_models:
            model_df = df_plot.filter(pl.col("model") == model)
            if model_df.height == 0:
                continue

            x_labels = model_df.get_column("point_short").to_list()
            pattern = model_patterns.get(model, "")

            for comp_key in ["rmse", "s", "u"]:
                if comp_key == "rmse":
                    col_name = f"RMSE_{axis_name}"
                else:
                    col_name = f"RMSE_{comp_key}_{axis_name}"

                if col_name not in model_df.columns:
                    continue

                y_data = model_df.get_column(col_name).to_list()

                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_data,
                        name=component_labels[comp_key],
                        offsetgroup=f"{model}_{comp_key}",
                        marker=dict(
                            color=component_colors[comp_key],
                            pattern_shape=pattern,
                            line=dict(width=0)  # Suppression de la bordure interne/externe pour un rendu net
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>%{{x}}</b> ({model})<br>"
                            + f"{component_labels[comp_key]}: "
                            + "%{y:.2f} mm<br>"
                            + "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    # --------------------------------------------------------
    # Layout and Styling
    # --------------------------------------------------------

    fig.update_layout(
        barmode="group",
        bargroupgap=0.15,
        height=1200,
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
        margin=dict(l=60, r=240, t=100, b=100),
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=ordered_points,
    )

    for axis_name, row_idx, col_idx in axes_config:
        fig.update_yaxes(
            title_text="RMSE (mm)",
            row=row_idx,
            col=col_idx,
        )
        fig.update_xaxes(
            title_text="Anatomical Landmarks",
            tickangle=45,
            row=row_idx,
            col=col_idx,
        )

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASET
# ============================================================

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
            "Willmott Global Error Decomposition — All Subjects & Trials",
            style={"textAlign": "center"},
        ),
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
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_points,
    selected_models,
):
    return plot_error_barplots(
        df=df,
        points=selected_points,
        models=selected_models,
        coordinate_system="anatomical",
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)