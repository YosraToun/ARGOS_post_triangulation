import generation_parquet
import generate_excel_comp_stat
from pathlib import Path
import generation_figure
import PlotConfig

if __name__ == "__main__":
    ### GENERATE PARQUET FILE
    path_data_kinematics = Path("./Kinematics_SB2026_all")
    path_to_export = Path("./figures")
    path_to_export.mkdir(exist_ok=True, parents=True)
    reference_name = "Marker_based"
    dof_hand = "with_2dof_hand"
    model_hpe = [
        "Synthpose",
        "SynthRTMPose",
        "Rtmpose",
    ]
    subject_laterality = {
        "S08": "R",
        "S09": "R",
        "S10": "R",
        "S11": "R",
        "S12": "R",
        "S13": "L",
        "S14": "R",
        "S15": "R",
        "S16": "R",
        "S17": "R",
        "S18": "R",
        "S19": "R",
        "S20": "R",
        "S21": "R",
        "S22": "R",
        "S23": "R",
        "S24": "R",}

    remove_before_after = {
        "16-comb-hair":(10, 10),
    }

    process_only_strong_side = True
    filename_output = path_to_export/"kinematic_errors_strong_side.parquet" if process_only_strong_side else path_to_export/"kinematic_errors.parquet"
    
    generation_parquet.generate_parquet(
        path_data_kinematics=path_data_kinematics,
        reference_name=reference_name,
        dof_hand=dof_hand,
        models_hpe=model_hpe,
        process_only_strong_side=process_only_strong_side,
        subject_laterality=subject_laterality,
        remove_before_after=remove_before_after,
        filename_output=filename_output
    )
    output_path_excel =  path_to_export/"kinematic_errors_summary.xlsx" if process_only_strong_side else path_to_export/"kinematic_errors_summary_all.xlsx"
    generate_excel_comp_stat.generate_excel_summary(
        df=generation_parquet.pl.read_parquet(filename_output),
        output_path=output_path_excel
    )

    ### GENERATE FIGURES
    data_to_plot = "error_deg" # abs_error_deg
    data_to_plot_label = "Error [deg]"
    path_to_figure = path_to_export / data_to_plot / "boxplot"
    path_to_figure_task = path_to_figure / "task"
    path_to_figure_dof = path_to_figure / "dof"


    path_to_figure.mkdir(exist_ok=True, parents=True)
    path_to_figure_task.mkdir(exist_ok=True, parents=True)
    path_to_figure_dof.mkdir(exist_ok=True, parents=True)
    path_to_data = filename_output


    dof_order = [
        "Humerus_RotX",
        "Humerus_RotZ",
        "Humerus_RotY",
        "Forearm_1_RotZ",
        "Forearm_2_RotY",
    ]

    Name_dof_order = {
        "Humerus_RotX": "Shoulder Adduction-Abduction",
        "Humerus_RotZ": "Shoulder Flexion-Extension",
        "Humerus_RotY": "Shoulder Internal-External Rotation",
        "Forearm_1_RotZ": "Elbow Flexion-Extension",
        "Forearm_2_RotY": "Forearm Pronation-Supination",
    }

    tasks_order = ["01-eat-yaourt",
                "03-Open_a_Bottle_and_Pour",
                "04-Hand_to_Head",
                "13-playdoe",
                "16-comb-hair",
                "10-reach_and_grasp_far",
                "26-Cymbals"]

    Name_tasks_order = {
        "01-eat-yaourt": "Eat Yaourt",
        "03-Open_a_Bottle_and_Pour": "Open a Bottle and Pour",
        "04-Hand_to_Head": "Hand to Head",
        "13-playdoe": "Playdoe",
        "16-comb-hair": "Comb Hair",
        "10-reach_and_grasp_far": "Reach and Grasp Far",
        "26-Cymbals": "Cymbals",
    }
    df = generation_parquet.pl.read_parquet(path_to_data)
    subjects_to_process = df.select("subject").unique().sort("subject").to_series().to_list()

    limit_boxplot_whiskers = (2.5, 97.5)
    config = PlotConfig.PlotConfig(
        data_to_plot=data_to_plot,
        data_to_plot_label="Error [deg]",
        whiskers=limit_boxplot_whiskers,
        dof_order=dof_order,
        dof_labels=Name_dof_order,
        task_order=tasks_order,
        task_labels=Name_tasks_order
    )
    config.resolve(df)
    generation_figure.all_subjects_tasks(
        subjects_to_process=subjects_to_process,
        path_to_data=path_to_data,
        output_dir=path_to_figure,
        config=config
    )
    generation_figure.by_task(path_to_data=path_to_data,
        output_dir=path_to_figure_task, config=config)

    generation_figure.by_dof(path_to_data=path_to_data,output_dir=path_to_figure_dof, config=config)