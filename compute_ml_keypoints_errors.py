from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame

# same as script of the same name in the folder "comparaison_3D_analysis" but add SynthposeRTM


def compare_absolute_file(
    path_marker_based: Path,
    path_synth: Path,
    path_RTM: Path,
    path_synth_RTM: Path,
    dict_Synth: dict,
    dict_RTM: dict,
    dict_Synth_RTM: dict,
) -> tuple[dict, dict, dict]:
    # compare the absolute position of the points in the marker based data and the marker less data
    acq_MB = ezc3d.c3d(str(path_marker_based))
    points_data_MB, points_name_MB, _ = snip.get_points_ezc3d(acq_MB)

    dict_error_Synth = {}
    dict_error_RTM = {}
    dict_error_Synth_RTM = {}

    if path_synth is not None and path_synth.exists():
        acq_Synth = ezc3d.c3d(str(path_synth))
        points_data_Synth, points_name_Synth, _ = snip.get_points_ezc3d(acq_Synth)
        for point_MB, point_synth in dict_Synth.items():
            if point_synth in points_name_Synth and point_MB in points_name_MB:
                error = calculate_error_absolute(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_Synth,
                    points_name=points_name_Synth,
                    point_name=point_synth,
                )
                dict_error_Synth[point_MB] = error

    if path_RTM is not None and path_RTM.exists():
        acq_RTM = ezc3d.c3d(str(path_RTM))
        points_data_RTM, points_name_RTM, _ = snip.get_points_ezc3d(acq_RTM)
        for point_MB, point_RTM in dict_RTM.items():
            if point_RTM in points_name_RTM and point_MB in points_name_MB:
                error = calculate_error_absolute(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_RTM,
                    points_name=points_name_RTM,
                    point_name=point_RTM,
                )
                dict_error_RTM[point_MB] = error

    if path_synth_RTM is not None and path_synth_RTM.exists():
        acq_Synth_RTM = ezc3d.c3d(str(path_synth_RTM))
        points_data_Synth_RTM, points_name_Synth_RTM, _ = snip.get_points_ezc3d(acq_Synth_RTM)
        for point_MB, point_synth_RTM in dict_Synth_RTM.items():
            if point_synth_RTM in points_name_Synth_RTM and point_MB in points_name_MB:
                error = calculate_error_absolute(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_Synth_RTM,
                    points_name=points_name_Synth_RTM,
                    point_name=point_synth_RTM,
                )
                dict_error_Synth_RTM[point_MB] = error

    return dict_error_Synth, dict_error_RTM, dict_error_Synth_RTM


def compare_anatomical_frame_file(
    path_synth: Path,
    path_RTM: Path,
    path_synth_RTM: Path,
    path_marker_based: Path,
    dict_Synth: dict,
    dict_RTM: dict,
    dict_Synth_RTM: dict,
    dict_frame_to_use: dict,
) -> tuple[dict, dict, dict]:
    # compare the position of the points in the marker based data and the marker less data in the anatomical frame
    acq_MB = ezc3d.c3d(str(path_marker_based))
    points_data_MB, points_name_MB, _ = snip.get_points_ezc3d(acq_MB)

    dict_error_Synth = {}
    dict_error_RTM = {}
    dict_error_Synth_RTM = {}

    if path_synth is not None and path_synth.exists():
        acq_Synth = ezc3d.c3d(str(path_synth))
        points_data_Synth, points_name_Synth, _ = snip.get_points_ezc3d(acq_Synth)
        for point_MB, point_synth in dict_Synth.items():
            if point_synth in points_name_Synth and point_MB in points_name_MB:
                frame_to_use = dict_frame_to_use[point_MB]
                side = point_MB[0]
                error = calculate_error_anatomical_frame(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_Synth,
                    points_name=points_name_Synth,
                    point_name=point_synth,
                    frame_to_use=frame_to_use,
                    side=side,
                )
                dict_error_Synth[point_MB] = error

    if path_RTM is not None and path_RTM.exists():
        acq_RTM = ezc3d.c3d(str(path_RTM))
        points_data_RTM, points_name_RTM, _ = snip.get_points_ezc3d(acq_RTM)
        for point_MB, point_RTM in dict_RTM.items():
            if point_RTM in points_name_RTM and point_MB in points_name_MB:
                frame_to_use = dict_frame_to_use[point_MB]
                side = point_MB[0]
                error = calculate_error_anatomical_frame(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_RTM,
                    points_name=points_name_RTM,
                    point_name=point_RTM,
                    frame_to_use=frame_to_use,
                    side=side,
                )
                dict_error_RTM[point_MB] = error

    if path_synth_RTM is not None and path_synth_RTM.exists():
        acq_Synth_RTM = ezc3d.c3d(str(path_synth_RTM))
        points_data_Synth_RTM, points_name_Synth_RTM, _ = snip.get_points_ezc3d(acq_Synth_RTM)
        for point_MB, point_synth_RTM in dict_Synth_RTM.items():
            if point_synth_RTM in points_name_Synth_RTM and point_MB in points_name_MB:
                frame_to_use = dict_frame_to_use[point_MB]
                side = point_MB[0]
                error = calculate_error_anatomical_frame(
                    points_data_ref=points_data_MB,
                    points_name_ref=points_name_MB,
                    point_name_ref=point_MB,
                    points_data=points_data_Synth_RTM,
                    points_name=points_name_Synth_RTM,
                    point_name=point_synth_RTM,
                    frame_to_use=frame_to_use,
                    side=side,
                )
                dict_error_Synth_RTM[point_MB] = error

    return dict_error_Synth, dict_error_RTM, dict_error_Synth_RTM


def calculate_error_absolute(
    points_data_ref: np.ndarray,
    points_name_ref: list,
    point_name_ref: str,
    points_data: np.ndarray,
    points_name: list,
    point_name: str,
):
    index_point = points_name.index(point_name)
    index_point_ref = points_name_ref.index(point_name_ref)

    point_coordinate = points_data[:, index_point, :]
    point_ref = points_data_ref[:, index_point_ref, :]

    error = point_coordinate - point_ref

    return error


def calculate_error_anatomical_frame(
    points_data_ref: np.ndarray,
    points_name_ref: list,
    point_name_ref: str,
    points_data: np.ndarray,
    points_name: list,
    point_name: str,
    frame_to_use: str,
    side="R",
):
    X_frame, Y_frame, Z_frame = extract_anatomical_frame(points_data_ref, points_name_ref, frame_to_use, side=side)

    index_point = points_name.index(point_name)
    index_point_ref = points_name_ref.index(point_name_ref)

    point_coordinate = points_data[:, index_point, :]
    point_ref = points_data_ref[:, index_point_ref, :]

    point_proj = project_point_onto_frame(point_coordinate, X_frame, Y_frame, Z_frame)
    point_ref_proj = project_point_onto_frame(point_ref, X_frame, Y_frame, Z_frame)

    error = point_proj - point_ref_proj

    return error


def error_dict_to_dataframe(
    error_dict: dict,
    subject: str,
    task: str,
    coordinate_system: str,
    dict_frame_to_use: dict,
    model: str,
) -> pl.DataFrame:
    dfs = []

    for point, error in error_dict.items():
        # error est supposé être de forme (3, n_frames)
        n_frames = error.shape[1]
        # remove L_ and R_ from the point name for one of the columns to ease the analysis
        point_short = point[2:] if point.startswith(("L_", "R_")) else point
        df = pl.DataFrame(
            {
                "subject": [subject] * n_frames,
                "task": [task] * n_frames,
                "coordinate_system": [coordinate_system] * n_frames,
                "model": [model] * n_frames,
                "point": [point] * n_frames,
                "point_short": [point_short] * n_frames,
                "frame": np.arange(n_frames),
                "frame_type": [dict_frame_to_use.get(point, "unknown")] * n_frames,
                "error_x": error[0, :],
                "error_y": error[1, :],
                "error_z": error[2, :],
            }
        ).with_columns(
            (pl.col("error_x") ** 2 + pl.col("error_y") ** 2 + pl.col("error_z") ** 2).sqrt().alias("error_norm")
        )

        dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs)


def project_point_onto_frame(point_data, X_frame, Y_frame, Z_frame):
    point_data_proj = np.array(
        [
            np.einsum("ij,ij->j", point_data, X_frame),
            np.einsum("ij,ij->j", point_data, Y_frame),
            np.einsum("ij,ij->j", point_data, Z_frame),
        ]
    )
    return point_data_proj


def main():
    path_data = Path(".\\data\\pre_processed")
    path_results_linear = Path(".\\results\\linear_kinematics")
    list_model_ML = ["SynthPose", "RTMPose", "SynthRTMPose"]
    path_RTM = path_data / "marker_less" / "RTMPose"
    path_synth = path_data / "marker_less" / "SynthPose"
    path_synth_RTM = path_data / "marker_less" / "SynthRTMPose"
    path_marker_based = path_data / "marker_based"

    # a dictionary allowing to define which point is compared with each point
    dict_comparaison_point_Synth = {
        "R_GH": "R_Shoulder",
        "L_GH": "L_Shoulder",
        "R_EJC": "R_Elbow",
        "L_EJC": "L_Elbow",
        "R_WJC": "R_Wrist",
        "L_WJC": "L_Wrist",
        "IJ": "sternum",
        "C7": "C7",
        "T10": "T11",
        "R_EL": "r_lelbow",
        "R_EM": "r_melbow",
        "L_EL": "l_lelbow",
        "L_EM": "l_melbow",
        "R_RS": "r_lwrist",
        "R_US": "r_mwrist",
        "L_RS": "l_lwrist",
        "L_US": "l_mwrist",
    }
    dict_comparaison_point_Synth_RTM = {
        "R_GH": "R_Shoulder",
        "L_GH": "L_Shoulder",
        "R_EJC": "R_Elbow",
        "L_EJC": "L_Elbow",
        "IJ": "sternum",
        "C7": "C7",
        "T10": "T11",
        "R_EL": "r_lelbow",
        "R_EM": "r_melbow",
        "L_EL": "l_lelbow",
        "L_EM": "l_melbow",
        "R_WJC": "R_base_hand",
        "L_WJC": "L_base_hand",
        "R_HM2": "R_MCP_index",
        "L_HM2": "L_MCP_index",
        "R_HM5": "R_MCP_little",
        "L_HM5": "L_MCP_little",
    }
    dict_comparaison_point_RTM = {
        "R_GH": "R_shoulder",
        "L_GH": "L_shoulder",
        "R_EJC": "R_elbow",
        "L_EJC": "L_elbow",
        "R_WJC": "R_base_hand",
        "L_WJC": "L_base_hand",
        "R_HM2": "R_MCP_index",
        "L_HM2": "L_MCP_index",
        "R_HM5": "R_MCP_little",
        "L_HM5": "L_MCP_little",
    }

    dict_frame_to_use = {
        "R_GH": "thorax",
        "L_GH": "thorax",
        "R_EJC": "humerus",
        "L_EJC": "humerus",
        "R_WJC": "forearm",
        "L_WJC": "forearm",
        "R_HM2": "hand",
        "L_HM2": "hand",
        "R_HM5": "hand",
        "L_HM5": "hand",
        "IJ": "thorax",
        "C7": "thorax",
        "T10": "thorax",
        "R_EL": "humerus",
        "R_EM": "humerus",
        "L_EL": "humerus",
        "L_EM": "humerus",
        "R_RS": "forearm",
        "R_US": "forearm",
        "L_RS": "forearm",
        "L_US": "forearm",
    }

    all_errors = []
    for subject in path_marker_based.iterdir():
        if not subject.is_dir():
            continue
        for task_file in subject.glob("*.c3d"):
            file_synth = path_synth / subject.name / task_file.name
            file_RTM = path_RTM / subject.name / task_file.name
            file_synth_RTM = path_synth_RTM / subject.name / task_file.name

            if file_synth.exists() or file_RTM.exists() or file_synth_RTM.exists():
                print(f"Comparing {task_file.name} for subject {subject.name}")
                dict_error_Synth, dict_error_RTM, dict_error_Synth_RTM = compare_absolute_file(
                    path_marker_based=task_file,
                    path_synth=file_synth,
                    path_RTM=file_RTM,
                    path_synth_RTM=file_synth_RTM,
                    dict_Synth=dict_comparaison_point_Synth,
                    dict_RTM=dict_comparaison_point_RTM,
                    dict_Synth_RTM=dict_comparaison_point_Synth_RTM,
                )
                dict_error_anat_Synth, dict_anat_error_RTM, dict_anat_error_Synth_RTM = compare_anatomical_frame_file(
                    path_synth=file_synth,
                    path_RTM=file_RTM,
                    path_synth_RTM=file_synth_RTM,
                    path_marker_based=task_file,
                    dict_Synth=dict_comparaison_point_Synth,
                    dict_RTM=dict_comparaison_point_RTM,
                    dict_Synth_RTM=dict_comparaison_point_Synth_RTM,
                    dict_frame_to_use=dict_frame_to_use,
                )

                dfs_trial = [
                    error_dict_to_dataframe(
                        dict_error_Synth,
                        subject.name,
                        task_file.stem,
                        "absolute",
                        dict_frame_to_use=dict_frame_to_use,
                        model="SynthPose",
                    ),
                    error_dict_to_dataframe(
                        dict_error_RTM,
                        subject.name,
                        task_file.stem,
                        "absolute",
                        dict_frame_to_use=dict_frame_to_use,
                        model="RTMPose",
                    ),
                    error_dict_to_dataframe(
                        dict_error_Synth_RTM,
                        subject.name,
                        task_file.stem,
                        "absolute",
                        dict_frame_to_use=dict_frame_to_use,
                        model="SynthRTMPose",
                    ),
                    error_dict_to_dataframe(
                        dict_error_anat_Synth,
                        subject.name,
                        task_file.stem,
                        "anatomical",
                        dict_frame_to_use=dict_frame_to_use,
                        model="SynthPose",
                    ),
                    error_dict_to_dataframe(
                        dict_anat_error_RTM,
                        subject.name,
                        task_file.stem,
                        "anatomical",
                        dict_frame_to_use=dict_frame_to_use,
                        model="RTMPose",
                    ),
                    error_dict_to_dataframe(
                        dict_anat_error_Synth_RTM,
                        subject.name,
                        task_file.stem,
                        "anatomical",
                        dict_frame_to_use=dict_frame_to_use,
                        model="SynthRTMPose",
                    ),
                ]
                all_errors.extend([df for df in dfs_trial if not df.is_empty()])

    if not all_errors:
        print("No error data generated.")
        return

    # Combine toutes les erreurs
    df_errors = pl.concat(all_errors).with_columns(
        pl.col(["error_x", "error_y", "error_z", "error_norm"]).fill_nan(None)
    )

    # Sauvegarde dans le dossier linear_kinematics
    path_results_linear.mkdir(parents=True, exist_ok=True)
    output_file = path_results_linear / "ml_keypoints_position_errors.parquet"
    df_errors.write_parquet(output_file)
    print(f"Errors saved to: {output_file}")
    print(df_errors)
    print(f"There is {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()