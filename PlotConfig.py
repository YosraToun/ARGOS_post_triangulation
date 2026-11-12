from dataclasses import dataclass
import polars as pl

@dataclass
class PlotConfig:
    """
    Configuration object used by the plotting functions defined in
    `generation_figure.py`.

    This class centralizes all parameters controlling how data extracted
    from the parquet file should be displayed, including variable selection,
    category ordering, axis labels, and boxplot whisker definitions.

    Parameters
    ----------
    data_to_plot : str
        Name of the column in the dataframe containing the metric to plot.
        Examples: `"error_deg", "error_rad","abs_error_deg".(correct name in generation_parquet.py)

    data_to_plot_label : str
        Display label for `data_to_plot`. This label is typically used
        for the y-axis title and figure annotations.

    whiskers : tuple[float, float]
        Lower and upper percentiles used to define the boxplot whiskers.

        Examples:
        - `(0.0, 100.0)` : display the full data range.
        - `(2.5, 97.5)` : display the central 95% of the distribution.
        - `(5.0, 95.0)` : display the central 90% of the distribution.

    dof_order : list[str] | None, optional
        Desired ordering of degrees of freedom (DOFs) in the plots.
        If None, the ordering is automatically inferred from the dataframe
        and sorted alphabetically.

    dof_labels : dict[str, str] | None, optional
        Mapping between internal DOF names and display names.

        Example:
        {
            "R_Humerus_1_RotX": "Humerus Flex/Ext",
            "R_Humerus_1_RotY": "Humerus Abd/Add"
        }

        If None, the original DOF names are used as labels.

    task_order : list[str] | None, optional
        Desired ordering of tasks in the generated figures.
        If None, tasks are automatically extracted from the dataframe
        and sorted alphabetically.

    task_labels : dict[str, str] | None, optional
        Mapping between task identifiers and display names.

        Example:
        {
        "01-eat-yaourt": "Eat Yaourt",
        "03-Open_a_Bottle_and_Pour": "Open a Bottle and Pour",
        "04-Hand_to_Head": "Hand to Head",
        "13-playdoe": "Playdoe",
        "16-comb-hair": "Comb Hair",
        "10-reach_and_grasp_far": "Reach and Grasp Far",
        "26-Cymbals": "Cymbals",
        }

        If None, task names are displayed as-is.

    Notes
    -----
    The `resolve()` method automatically fills any missing configuration
    values using information available in the provided dataframe:

    - `task_order` is inferred from the unique values of the `task` column.
    - `dof_order` is inferred from the unique values of the `dof` column.
    - `task_labels` defaults to an identity mapping.
    - `dof_labels` defaults to an identity mapping.
    - `whiskers` defaults to `(0.0, 100.0)` if not specified.

    This ensures that plotting functions always receive a complete and
    consistent configuration object regardless of which fields were
    explicitly provided by the user.
    """
    data_to_plot: str
    data_to_plot_label: str
    whiskers: tuple[float, float]
    dof_order: list[str] | None = None
    dof_labels: dict[str, str] | None = None
    task_order: list[str] | None = None
    task_labels: dict[str, str] | None = None

    def resolve(self, df: pl.DataFrame):

        if self.task_order is None:
            self.task_order = (
                df.select("task")
                .unique()
                .sort("task")
                .to_series()
                .to_list()
            )

        if self.dof_order is None:
            self.dof_order = (
                df.select("dof")
                .unique()
                .sort("dof")
                .to_series()
                .to_list()
            )

        if self.task_labels is None:
            self.task_labels = {
                task: task
                for task in self.task_order
            }

        if self.dof_labels is None:
            self.dof_labels = {
                dof: dof
                for dof in self.dof_order
            }

        if self.whiskers is None:
            self.whiskers = (0.0, 100.0)