from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_to_data = Path(r".\data\pre_processed\Robeson_3Dtrajectories_errors.parquet")

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
# PLOT FUNCTION (DYNAMIC GRID FOR 3 OR FEWER POINTS)
# ============================================================


def plot_error_percentages(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):

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

    cols = df_plot.columns
    col_bx = "MAE_b_X" if "MAE_b_X" in cols else "MAE_b_x"
    col_px = "MAE_p_X" if "MAE_p_X" in cols else "MAE_p_x"
    col_ux = "MAE_u_X" if "MAE_u_X" in cols else "MAE_u_x"

    col_by = "MAE_b_Y" if "MAE_b_Y" in cols else "MAE_b_y"
    col_py = "MAE_p_Y" if "MAE_p_Y" in cols else "MAE_p_y"
    col_uy = "MAE_u_Y" if "MAE_u_Y" in cols else "MAE_u_y"

    col_bz = "MAE_b_Z" if "MAE_b_Z" in cols else "MAE_b_z"
    col_pz = "MAE_p_Z" if "MAE_p_Z" in cols else "MAE_p_z"
    col_uz = "MAE_u_Z" if "MAE_u_Z" in cols else "MAE_u_z"

    # --------------------------------------------------------
    # Aggregate and Compute Percentages per Point and Model
    # --------------------------------------------------------

    agg_df = (
        df_plot
        .group_by(["point_short", "model"])
        .agg(
            pl.col(col_bx).mean().fill_null(0).alias("b_x"),
            pl.col(col_px).mean().fill_null(0).alias("p_x"),
            pl.col(col_ux).mean().fill_null(0).alias("u_x"),
            pl.col(col_by).mean().fill_null(0).alias("b_y"),
            pl.col(col_py).mean().fill_null(0).alias("p_y"),
            pl.col(col_uy).mean().fill_null(0).alias("u_y"),
            pl.col(col_bz).mean().fill_null(0).alias("b_z"),
            pl.col(col_pz).mean().fill_null(0).alias("p_z"),
            pl.col(col_uz).mean().fill_null(0).alias("u_z"),
        )
        .with_columns([
            (pl.col("b_x") + pl.col("p_x") + pl.col("u_x")).alias("total_x"),
            (pl.col("b_y") + pl.col("p_y") + pl.col("u_y")).alias("total_y"),
            (pl.col("b_z") + pl.col("p_z") + pl.col("u_z")).alias("total_z"),
        ])
        .with_columns([
            pl.when(pl.col("total_x") > 0).then(pl.col("b_x") / pl.col("total_x") * 100).otherwise(0).alias("pct_b_x"),
            pl.when(pl.col("total_x") > 0).then(pl.col("p_x") / pl.col("total_x") * 100).otherwise(0).alias("pct_p_x"),
            pl.when(pl.col("total_x") > 0).then(pl.col("u_x") / pl.col("total_x") * 100).otherwise(0).alias("pct_u_x"),

            pl.when(pl.col("total_y") > 0).then(pl.col("b_y") / pl.col("total_y") * 100).otherwise(0).alias("pct_b_y"),
            pl.when(pl.col("total_y") > 0).then(pl.col("p_y") / pl.col("total_y") * 100).otherwise(0).alias("pct_p_y"),
            pl.when(pl.col("total_y") > 0).then(pl.col("u_y") / pl.col("total_y") * 100).otherwise(0).alias("pct_u_y"),

            pl.when(pl.col("total_z") > 0).then(pl.col("b_z") / pl.col("total_z") * 100).otherwise(0).alias("pct_b_z"),
            pl.when(pl.col("total_z") > 0).then(pl.col("p_z") / pl.col("total_z") * 100).otherwise(0).alias("pct_p_z"),
            pl.when(pl.col("total_z") > 0).then(pl.col("u_z") / pl.col("total_z") * 100).otherwise(0).alias("pct_u_z"),
        ])
    )

    # Sort according to POINT_ORDER
    mapping_df = pl.DataFrame({
        "point_short": POINT_ORDER,
        "sort_idx": range(len(POINT_ORDER))
    })

    agg_df = (
        agg_df
        .join(mapping_df, on="point_short", how="inner")
        .sort(["sort_idx", "model"])
    )

    # --------------------------------------------------------
    # Dynamic Subplot Setup (Compact if <= 3 points)
    # --------------------------------------------------------

    current_points = agg_df.get_column("point_short").unique().to_list()
    # Same category order on every subplot (follows POINT_ORDER)
    ordered_points = [p for p in POINT_ORDER if p in current_points]
    is_compact = len(ordered_points) <= 3

    titles = [
        "X Axis",
        "Y Axis",
        "Z Axis",
    ]

    if is_compact:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=titles,
            vertical_spacing=0.22,
            horizontal_spacing=0.1,
        )
        axes_config = [
            ("x", 1, 1),
            ("y", 1, 2),
            ("z", 2, 1),
        ]
        fig_height = 900
    else:
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
        fig_height = 1200

    component_colors = {
        "b": "#d62728",     # Systematic Bias (Red)
        "p": "#ff7f0e",     # Proportional Error (Orange)
        "u": "#1f77b4",     # Unsystematic Error (Blue)
    }

    component_labels = {
        "b": "Systematic Bias",
        "p": "Proportional Error",
        "u": "Unsystematic Error",
    }

    model_patterns = {
        "SynthPose": "",
        "RTMPose": "/",
    }

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
                legendgrouptitle_text="<b>Error Components</b>" if key == "b" else None,
                showlegend=True
            ),
            row=1, col=1
        )

    for model_name, pattern in model_patterns.items():
        legend_name = f"{model_name} (Hatched)" if model_name == "RTMPose" else model_name
        fig.add_trace(
            go.Bar(
                x=[anchor_x], y=[None],
                offsetgroup=model_name,  # same slot as the real bars -> same bar width on all subplots
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

    available_models = agg_df.get_column("model").unique().to_list()

    for axis_name, row_idx, col_idx in axes_config:
        for model in available_models:
            model_df = agg_df.filter(pl.col("model") == model)
            if model_df.height == 0:
                continue

            x_labels = model_df.get_column("point_short").to_list()
            pattern = model_patterns.get(model, "")

            for comp_key in ["b", "p", "u"]:
                y_data = model_df.get_column(f"pct_{comp_key}_{axis_name}").to_list()
                mm_data = model_df.get_column(f"{comp_key}_{axis_name}").to_list()
                total_mm_data = model_df.get_column(f"total_{axis_name}").to_list()

                custom_data = list(zip(mm_data, total_mm_data))

                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_data,
                        customdata=custom_data,
                        name=component_labels[comp_key],
                        offsetgroup=model,
                        marker=dict(
                            color=component_colors[comp_key],
                            pattern_shape=pattern,
                            line=dict(color="black", width=0.5)
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>%{{x}}</b> ({model})<br>"
                            + f"{component_labels[comp_key]}: "
                            + "%{y:.1f}% (%{customdata[0]:.2f} mm)<br>"
                            + "Total Error: %{customdata[1]:.2f} mm"
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
        barmode="stack",
        bargroupgap=0.25,
        height=fig_height,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=16),
            title_font=dict(size=18),
            tracegroupgap=30
        ),
        margin=dict(l=60, r=240, t=100, b=100),
    )

    # Force all X axes to be categorical with the same fixed order
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=ordered_points,
    )

    for axis_name, row_idx, col_idx in axes_config:
        fig.update_yaxes(
            title_text="Error Proportion (%)",
            range=[0, 100],
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

all_tasks = df.select("task").unique().sort("task").to_series().to_list()
all_subjects = df.select("subject").unique().sort("subject").to_series().to_list()
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
            "Robeson Error Percentage Breakdown",
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
        html.H3("Subjects"),
        dcc.Dropdown(
            id="subject-selector",
            options=[{"label": subject, "value": subject} for subject in all_subjects],
            value=all_subjects,
            multi=True,
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
    Input("task-selector", "value"),
    Input("subject-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_tasks,
    selected_subjects,
    selected_points,
    selected_models,
):
    return plot_error_percentages(
        df=df,
        tasks=selected_tasks,
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