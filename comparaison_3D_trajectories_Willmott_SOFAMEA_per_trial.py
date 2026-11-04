from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame

#ALL subjects pooled together for each task 
#Equation Willmott (1981, 1982) for RMSE decomposition into systematic and unsystematic components

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

    # Si le nombre de points valides est insuffisant pour la régression, retourner 0.0
    n = len(obs)
    if n < 2:
        return {"RMSE": 0.0, "RMSE_s": 0.0, "RMSE_u": 0.0}

    # Régression OLS globale pour cet essai (tous sujets confondus)
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


def project_point_onto_frame(point_data, X_frame, Y_frame, Z_frame):
    point_data_proj = np.array(
        [
            np.einsum("ij,ij->j", point_data, X_frame),
            np.einsum("ij,ij->j", point_data, Y_frame),
            np.einsum("ij,ij->j", point_data, Z_frame),
        ]
    )
    return point_data_proj


def collect_file_data(
    path_MB: Path,
    path_model: Path,
    dict_mapping: dict,
    dict_frame_to_use: dict,
    model_name: str,
    task_name: str,
    storage: dict,
):
    acq_MB = ezc3d.c3d(str(path_MB))
    acq_model = ezc3d.c3d(str(path_model))

    points_data_MB, points_name_MB, _ = snip.get_points_ezc3d(acq_MB)
    points_data_model, points_name_model, _ = snip.get_points_ezc3d(acq_model)

    for point_MB, point_model in dict_mapping.items():
        if point_model not in points_name_model or point_MB not in points_name_MB:
            continue

        idx_ref = points_name_MB.index(point_MB)
        idx_mod = points_name_model.index(point_model)

        ref_coords = points_data_MB[:, idx_ref, :]       # Shape (3, n_frames)
        mod_coords = points_data_model[:, idx_mod, :]    # Shape (3, n_frames)

        # 1. Système de coordonnées absolues (groupé par task)
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            obs = ref_coords[axis_idx, :]
            pre = mod_coords[axis_idx, :]
            valid = ~np.isnan(obs) & ~np.isnan(pre)
            if np.any(valid):
                key = ("absolute", model_name, point_MB, task_name, axis_name)
                if key not in storage:
                    storage[key] = {"obs": [], "pre": []}
                storage[key]["obs"].append(obs[valid])
                storage[key]["pre"].append(pre[valid])

        # 2. Système de coordonnées anatomiques (groupé par task)
        if point_MB in dict_frame_to_use:
            frame_to_use = dict_frame_to_use[point_MB]
            side = point_MB[0]
            try:
                X_frame, Y_frame, Z_frame = extract_anatomical_frame(points_data_MB, points_name_MB, frame_to_use, side=side)
                ref_proj = project_point_onto_frame(ref_coords, X_frame, Y_frame, Z_frame)
                mod_proj = project_point_onto_frame(mod_coords, X_frame, Y_frame, Z_frame)

                for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                    obs = ref_proj[axis_idx, :]
                    pre = mod_proj[axis_idx, :]
                    valid = ~np.isnan(obs) & ~np.isnan(pre)
                    if np.any(valid):
                        key = ("anatomical", model_name, point_MB, task_name, axis_name)
                        if key not in storage:
                            storage[key] = {"obs": [], "pre": []}
                        storage[key]["obs"].append(obs[valid])
                        storage[key]["pre"].append(pre[valid])
            except Exception:
                pass


def build_task_pooled_dataframe(storage: dict, dict_frame_to_use: dict) -> pl.DataFrame:
    results_dict = {}

    for (coord_sys, model, point, task, axis), data in storage.items():
        obs_pooled = np.concatenate(data["obs"])
        pre_pooled = np.concatenate(data["pre"])

        decomp = rmse_decomp(obs_pooled, pre_pooled)

        if (coord_sys, model, point, task) not in results_dict:
            results_dict[(coord_sys, model, point, task)] = {}
        
        results_dict[(coord_sys, model, point, task)][axis] = decomp

    dfs = []
    for (coord_sys, model, point, task), axes_metrics in results_dict.items():
        point_short = point[2:] if point.startswith(("L_", "R_")) else point
        frame_type = dict_frame_to_use.get(point, "unknown") if coord_sys == "anatomical" else "absolute"

        row_data = {
            "task": [task],
            "coordinate_system": [coord_sys],
            "model": [model],
            "point": [point],
            "point_short": [point_short],
            "frame_type": [frame_type],
        }

        for axis in ["x", "y", "z"]:
            metrics = axes_metrics.get(axis, {"RMSE": 0.0, "RMSE_s": 0.0, "RMSE_u": 0.0})
            row_data[f"RMSE_{axis}"] = [metrics["RMSE"]]
            row_data[f"RMSE_s_{axis}"] = [metrics["RMSE_s"]]
            row_data[f"RMSE_u_{axis}"] = [metrics["RMSE_u"]]

        dfs.append(pl.DataFrame(row_data))

    if not dfs:
        return pl.DataFrame()
    return pl.concat(dfs)


def main():
    path_data = Path(r".\data\pre_processed")
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
    
    # Dictionnaire pour stocker et grouper les données par essai (task)
    storage = {}

    for subject in path_marker_based.iterdir():
        if not subject.is_dir():
            continue
        for task_file in subject.glob("*.c3d"):
            path_mb_file = subject / task_file.name
            path_synth_file = path_synth / subject.name / task_file.name
            path_rtm_file = path_RTM / subject.name / task_file.name

            if path_synth_file.exists() and path_rtm_file.exists():
                task_name = task_file.stem
                print(f"Collecting data for task '{task_name}' from subject {subject.name}")
                
                # Collecter SynthPose pour cet essai (tous sujets confondus par task_name)
                collect_file_data(
                    path_mb_file, path_synth_file,
                    dict_comparaison_point_Synth, dict_frame_to_use,
                    model_name="SynthPose", task_name=task_name, storage=storage
                )
                
                # Collecter RTMPose pour cet essai
                collect_file_data(
                    path_mb_file, path_rtm_file,
                    dict_comparaison_point_RTM, dict_frame_to_use,
                    model_name="RTMPose", task_name=task_name, storage=storage
                )

    print("Computing task-pooled regression and RMSE decomposition across all subjects...")
    df_errors = build_task_pooled_dataframe(storage, dict_frame_to_use)

    output_file = path_data / "Willmott_3Dtrajectories_errors_per_trial.parquet"
    df_errors.write_parquet(output_file)
    print(f"Task-pooled errors saved to: {output_file}")
    print(df_errors)
    print(f"There are {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()