from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame


def mae_decomp(obs, pre):
    """Decompose MAE into systematic and unsystematic components
    (Robeson and Willmott, PLOS ONE)
    """
    obs = np.asarray(obs, dtype=float)
    pre = np.asarray(pre, dtype=float)

    # Filter out any indices where either obs or pre is NaN
    valid_mask = ~np.isnan(obs) & ~np.isnan(pre)
    obs = obs[valid_mask]
    pre = pre[valid_mask]

    # If not enough valid frames remain for regression, return 0.0
    n = len(obs)
    if n < 2:
        return {"MAE_b": 0.0, "MAE_p": 0.0, "MAE_u": 0.0}

    # Setup
    abserr = np.abs(pre - obs)
    mbe = np.mean(pre) - np.mean(obs)
    pre_prime = pre - mbe

    # OLS regression to get phat_prime (degree 1 polyfit returns [slope, intercept])
    slope, intercept = np.polyfit(obs, pre_prime, 1)
    phat_prime = intercept + slope * obs

    # Form weights
    b = np.abs(mbe)
    p = np.abs(phat_prime - obs)
    u = np.abs(pre_prime - phat_prime)
    denom = b + p + u

    # Prevent division by zero if all errors are zero
    if np.all(denom == 0):
        return {"MAE_b": 0.0, "MAE_p": 0.0, "MAE_u": 0.0}

    bias = b / denom
    prop = p / denom
    unsys = u / denom

    # Components of MAE
    maeb = np.sum(bias * abserr) / n
    maep = np.sum(prop * abserr) / n
    maeu = np.sum(unsys * abserr) / n

    return {"MAE_b": maeb, "MAE_p": maep, "MAE_u": maeu}


def compare_absolute_file(
    path_marker_based: Path, path_synth: Path, path_RTM: Path, dict_Synth: dict, dict_RTM: dict
) -> tuple[dict, dict]:
    acq_Synth = ezc3d.c3d(str(path_synth))
    acq_RTM = ezc3d.c3d(str(path_RTM))
    acq_MB = ezc3d.c3d(str(path_marker_based))

    points_data_Synth, points_name_Synth, points_Synth = snip.get_points_ezc3d(acq_Synth)
    points_data_RTM, points_name_RTM, points_RTM = snip.get_points_ezc3d(acq_RTM)
    points_data_MB, points_name_MB, points_MB = snip.get_points_ezc3d(acq_MB)
    
    dict_error_Synth = {}
    dict_error_RTM = {}
    
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

    return dict_error_Synth, dict_error_RTM


def compare_anatomical_frame_file(
    path_synth: Path, path_RTM: Path, path_marker_based: Path, dict_Synth: dict, dict_RTM: dict, dict_frame_to_use: dict
) -> tuple[dict, dict]:
    acq_Synth = ezc3d.c3d(str(path_synth))
    acq_RTM = ezc3d.c3d(str(path_RTM))
    acq_MB = ezc3d.c3d(str(path_marker_based))
    
    points_data_Synth, points_name_Synth, points_Synth = snip.get_points_ezc3d(acq_Synth)
    points_data_RTM, points_name_RTM, points_RTM = snip.get_points_ezc3d(acq_RTM)
    points_data_MB, points_name_MB, points_MB = snip.get_points_ezc3d(acq_MB)
    
    dict_error_Synth = {}
    dict_error_RTM = {}
    
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

    return dict_error_Synth, dict_error_RTM


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

    # Compute Robeson MAE decomposition for X, Y, and Z axes across frames
    decomp_x = mae_decomp(point_ref[0, :], point_coordinate[0, :])
    decomp_y = mae_decomp(point_ref[1, :], point_coordinate[1, :])
    decomp_z = mae_decomp(point_ref[2, :], point_coordinate[2, :])

    return {"x": decomp_x, "y": decomp_y, "z": decomp_z}


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

    # Compute Robeson MAE decomposition on projected coordinates for X, Y, and Z axes
    decomp_x = mae_decomp(point_ref_proj[0, :], point_proj[0, :])
    decomp_y = mae_decomp(point_ref_proj[1, :], point_proj[1, :])
    decomp_z = mae_decomp(point_ref_proj[2, :], point_proj[2, :])

    return {"x": decomp_x, "y": decomp_y, "z": decomp_z}


def error_dict_to_dataframe(
    error_dict: dict,
    subject: str,
    task: str,
    coordinate_system: str,
    dict_frame_to_use: dict,
    model: str,
) -> pl.DataFrame:
    dfs = []

    for point, metrics in error_dict.items():
        point_short = point[2:] if point.startswith(("L_", "R_")) else point
        
        df = pl.DataFrame(
            {
                "subject": [subject],
                "task": [task],
                "coordinate_system": [coordinate_system],
                "model": [model],
                "point": [point],
                "point_short": [point_short],
                "frame_type": [dict_frame_to_use.get(point, "unknown")],
                
                # X axis decomposition metrics
                "MAE_b_x": [metrics["x"]["MAE_b"]],
                "MAE_p_x": [metrics["x"]["MAE_p"]],
                "MAE_u_x": [metrics["x"]["MAE_u"]],
                
                # Y axis decomposition metrics
                "MAE_b_y": [metrics["y"]["MAE_b"]],
                "MAE_p_y": [metrics["y"]["MAE_p"]],
                "MAE_u_y": [metrics["y"]["MAE_u"]],
                
                # Z axis decomposition metrics
                "MAE_b_z": [metrics["z"]["MAE_b"]],
                "MAE_p_z": [metrics["z"]["MAE_p"]],
                "MAE_u_z": [metrics["z"]["MAE_u"]],
            }
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
    path_RTM = path_data / "marker_less" / "RTMPose"
    path_synth = path_data / "marker_less" / "SynthPose"
    path_marker_based = path_data / "marker_based"

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
        for task_file in (subject).glob("*.c3d"):
            if (path_synth / subject.name / task_file.name).exists() and (
                path_RTM / subject.name / task_file.name
            ).exists():
                print(f"Comparing {task_file.name} for subject {subject.name}")
                dict_error_Synth, dict_error_RTM = compare_absolute_file(
                    path_marker_based / subject.name / task_file.name,
                    path_synth / subject.name / task_file.name,
                    path_RTM / subject.name / task_file.name,
                    dict_comparaison_point_Synth,
                    dict_comparaison_point_RTM,
                )
                dict_error_anat_Synth, dict_anat_error_RTM = compare_anatomical_frame_file(
                    path_synth / subject.name / task_file.name,
                    path_RTM / subject.name / task_file.name,
                    path_marker_based / subject.name / task_file.name,
                    dict_comparaison_point_Synth,
                    dict_comparaison_point_RTM,
                    dict_frame_to_use,
                )

                df_error_Synth = error_dict_to_dataframe(
                    dict_error_Synth,
                    subject.name,
                    task_file.stem,
                    "absolute",
                    dict_frame_to_use=dict_frame_to_use,
                    model="SynthPose",
                )
                df_error_RTM = error_dict_to_dataframe(
                    dict_error_RTM,
                    subject.name,
                    task_file.stem,
                    "absolute",
                    dict_frame_to_use=dict_frame_to_use,
                    model="RTMPose",
                )
                df_error_anat_Synth = error_dict_to_dataframe(
                    dict_error_anat_Synth,
                    subject.name,
                    task_file.stem,
                    "anatomical",
                    dict_frame_to_use=dict_frame_to_use,
                    model="SynthPose",
                )
                df_error_anat_RTM = error_dict_to_dataframe(
                    dict_anat_error_RTM,
                    subject.name,
                    task_file.stem,
                    "anatomical",
                    dict_frame_to_use=dict_frame_to_use,
                    model="RTMPose",
                )

                all_errors.extend([df_error_Synth, df_error_RTM, df_error_anat_Synth, df_error_anat_RTM])

    df_errors = pl.concat(all_errors)
    output_file = path_data / "Robeson_3Dtrajectories_errors.parquet"
    df_errors.write_parquet(output_file)
    print(f"Errors saved to: {output_file}")
    print(df_errors)
    print(f"There are {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()