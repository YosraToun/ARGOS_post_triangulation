from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame


def rmse_decomp(obs, pre):
    """Décompose la MSE en composantes systématique et unsystématique 
    selon Willmott (1981, 1982) et retourne les racines carrées (RMSE) en millimètres.
    """
    obs = np.asarray(obs, dtype=float)
    pre = np.asarray(pre, dtype=float)

    # Filtrer les valeurs NaN
    valid_mask = ~np.isnan(obs) & ~np.isnan(pre)
    obs = obs[valid_mask]
    pre = pre[valid_mask]

    # Si le nombre de frames valides est insuffisant pour la régression, retourner 0.0
    n = len(obs)
    if n < 2:
        return {"RMSE": 0.0, "RMSE_s": 0.0, "RMSE_u": 0.0}

    # Régression OLS des prédictions (pre) sur les observations (obs) : pre_hat = intercept + slope * obs
    slope, intercept = np.polyfit(obs, pre, 1)
    pre_hat = intercept + slope * obs

    # Calcul des composantes de la MSE puis passage à la racine carrée (RMSE)
    mse = np.sum((pre - obs) ** 2) / n
    mses = np.sum((pre_hat - obs) ** 2) / n
    mseu = np.sum((pre - pre_hat) ** 2) / n

    return {
        "RMSE": np.sqrt(mse),
        "RMSE_s": np.sqrt(mses),
        "RMSE_u": np.sqrt(mseu)
    }


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

    # Calcul de la décomposition RMSE de Willmott pour les axes X, Y et Z
    decomp_x = rmse_decomp(point_ref[0, :], point_coordinate[0, :])
    decomp_y = rmse_decomp(point_ref[1, :], point_coordinate[1, :])
    decomp_z = rmse_decomp(point_ref[2, :], point_coordinate[2, :])

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

    # Calcul de la décomposition RMSE de Willmott sur les coordonnées projetées
    decomp_x = rmse_decomp(point_ref_proj[0, :], point_proj[0, :])
    decomp_y = rmse_decomp(point_ref_proj[1, :], point_proj[1, :])
    decomp_z = rmse_decomp(point_ref_proj[2, :], point_proj[2, :])

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
                
                # Métriques RMSE pour l'axe X
                "RMSE_x": [metrics["x"]["RMSE"]],
                "RMSE_s_x": [metrics["x"]["RMSE_s"]],
                "RMSE_u_x": [metrics["x"]["RMSE_u"]],
                
                # Métriques RMSE pour l'axe Y
                "RMSE_y": [metrics["y"]["RMSE"]],
                "RMSE_s_y": [metrics["y"]["RMSE_s"]],
                "RMSE_u_y": [metrics["y"]["RMSE_u"]],
                
                # Métriques RMSE pour l'axe Z
                "RMSE_z": [metrics["z"]["RMSE"]],
                "RMSE_s_z": [metrics["z"]["RMSE_s"]],
                "RMSE_u_z": [metrics["z"]["RMSE_u"]],
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
    output_file = path_data / "Willmott_3Dtrajectories_errors.parquet"
    df_errors.write_parquet(output_file)
    print(f"Errors saved to: {output_file}")
    print(df_errors)
    print(f"There are {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()