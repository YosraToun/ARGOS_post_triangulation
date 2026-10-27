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

# Label of the extra bar summarising all selected landmarks
SUMMARY_LABEL = "ALL"


# ============================================================
# SHARED FILTER
# ============================================================


def filter_data(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
) -> pl.DataFrame:
    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    return df_plot


# ============================================================
# BARPLOT FUNCTION (DYNAMIC GRID FOR 3 OR FEWER POINTS)
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

    df_plot = filter_data(df, subjects, tasks, points, models, coordinate_system)

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

    per_point_df = (
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
    )

    # Extra "ALL" bar: mean over the selected landmarks (equal weight per landmark), per model
    value_cols = ["b_x", "p_x", "u_x", "b_y", "p_y", "u_y", "b_z", "p_z", "u_z"]
    summary_df = (
        per_point_df
        .group_by("model")
        .agg([pl.col(c).mean() for c in value_cols])
        .with_columns(pl.lit(SUMMARY_LABEL).alias("point_short"))
        .select(per_point_df.columns)
    )

    agg_df = (
        pl.concat([per_point_df, summary_df])
        .with_columns([
            (pl.col(f"b_{a}") + pl.col(f"p_{a}") + pl.col(f"u_{a}")).alias(f"total_{a}")
            for a in ("x", "y", "z")
        ])
        .with_columns([
            pl.when(pl.col(f"total_{a}") > 0)
            .then(pl.col(f"{c}_{a}") / pl.col(f"total_{a}") * 100)
            .otherwise(0)
            .alias(f"pct_{c}_{a}")
            for a in ("x", "y", "z")
            for c in ("b", "p", "u")
        ])
    )

    # Sort according to POINT_ORDER, with the summary bar last
    mapping_df = pl.DataFrame({
        "point_short": POINT_ORDER + [SUMMARY_LABEL],
        "sort_idx": range(len(POINT_ORDER) + 1),
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
    landmark_points = [p for p in POINT_ORDER if p in current_points]
    ordered_points = landmark_points + [SUMMARY_LABEL]  # summary bar always last
    is_compact = len(landmark_points) <= 3

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
# BOXPLOT FUNCTION
# ============================================================


def plot_error_quantiles(
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

    df_plot = filter_data(df, subjects, tasks, points, models, coordinate_system)

    # Compute Total MAE per axis
    df_plot = df_plot.with_columns([
        (pl.col("MAE_b_x") + pl.col("MAE_p_x") + pl.col("MAE_u_x")).alias("MAE_total_x"),
        (pl.col("MAE_b_y") + pl.col("MAE_p_y") + pl.col("MAE_u_y")).alias("MAE_total_y"),
        (pl.col("MAE_b_z") + pl.col("MAE_p_z") + pl.col("MAE_u_z")).alias("MAE_total_z"),
    ])

    # --------------------------------------------------------
    # Unpivot Robeson Decomposition Columns
    # --------------------------------------------------------

    robeson_columns = [
        "MAE_b_x", "MAE_p_x", "MAE_u_x",
        "MAE_b_y", "MAE_p_y", "MAE_u_y",
        "MAE_b_z", "MAE_p_z", "MAE_u_z",
        "MAE_total_x", "MAE_total_y", "MAE_total_z",
    ]

    df_unpivoted = (
        df_plot.unpivot(
            index=["point_short", "model", "subject", "task"],
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

    # Color mappings for the 4 components
    component_colors = {
        "b": "#d62728",     # Systematic Bias (Red)
        "p": "#ff7f0e",     # Proportional Error (Orange)
        "u": "#1f77b4",     # Unsystematic Error (Blue)
        "total": "#2ca02c", # Total MAE (Green)
    }

    # Model Line Style mapping
    model_dash = {
        "SynthPose": "solid",
        "RTMPose": "dash",
    }

    # --------------------------------------------------------
    # Add Grouped & Cleanly Separated Legend Traces
    # --------------------------------------------------------
    # 1. Error Components Group
    component_labels = {
        "b": "Systematic Bias",
        "p": "Proportional Error",
        "u": "Unsystematic Error",
        "total": "Total MAE",
    }
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

    # 2. Models Group (Line Styles)
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

            components = ["b", "p", "u", "total"]

            for comp_index, component in enumerate(components):
                comp_data = model_data.filter(pl.col("component") == component)
                color = component_colors[component]

                # Offset calculation for 2 models x 4 components = 8 items per point
                offset = (model_index * 4 + comp_index - 3.5) * 0.05

                for point_index, point in enumerate(available_points):
                    q = comp_data.filter(pl.col("point_short") == point)
                    if q.height == 0:
                        continue

                    q05, q20, median, q70, q95 = q["q05"][0], q["q20"][0], q["median"][0], q["q70"][0], q["q95"][0]
                    x = point_index + offset

                    # Draw whisker, box, and median with dash_style
                    fig.add_trace(go.Scatter(x=[x, x], y=[q05, q95], mode="lines", line=dict(color=color, width=1.5, dash=dash_style), showlegend=False, hoverinfo="skip"), row=row_idx, col=1)
                    fig.add_shape(type="rect", x0=x - 0.022, x1=x + 0.022, y0=q20, y1=q70, line=dict(color=color, width=1.5, dash=dash_style), fillcolor=color, opacity=0.3, row=row_idx, col=1)
                    fig.add_trace(go.Scatter(x=[x - 0.022, x + 0.022], y=[median, median], mode="lines", line=dict(color=color, width=2, dash=dash_style), showlegend=False, hoverinfo="skip"), row=row_idx, col=1)

    # --------------------------------------------------------
    # X and Y Axis Styling
    # --------------------------------------------------------
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
            title_text="Error (mm)",
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        height=1300,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",          # Vertical stacking so items are on different lines
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,                   # Placed cleanly on the right side
            font=dict(size=14),       # Bigger font size
            tracegroupgap=25          # Space between Error Components and Models groups
        ),
        margin=dict(l=60, r=220, t=100, b=100),  # Extra right margin for the vertical legend
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

PLOT_TYPES = ["Boxplot", "Barplot"]


# ============================================================
# DASH APPLICATION LAYOUT
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Robeson Error Decomposition",
            style={"textAlign": "center"},
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
        html.H3("Plot type"),
        dcc.Dropdown(
            id="plot-type-selector",
            options=[{"label": p, "value": p} for p in PLOT_TYPES],
            value="Barplot",
            multi=False,
            clearable=False,
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
    Output("error-graph", "style"),
    Input("plot-type-selector", "value"),
    Input("task-selector", "value"),
    Input("subject-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_plot_type,
    selected_tasks,
    selected_subjects,
    selected_points,
    selected_models,
):
    plot_function = (
        plot_error_quantiles if selected_plot_type == "Boxplot" else plot_error_percentages
    )

    fig = plot_function(
        df=df,
        tasks=selected_tasks,
        subjects=selected_subjects,
        points=selected_points,
        models=selected_models,
        coordinate_system="anatomical",
    )

    # Match the graph container height to the figure height
    return fig, {"height": f"{fig.layout.height + 50}px"}


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)