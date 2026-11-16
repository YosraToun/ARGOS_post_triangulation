from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame


def mae_decomp(obs, pre):
    """Decompose MAE into systematic (bias), proportional, and unsystematic components
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

        # 1. Absolute coordinate system
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            obs = ref_coords[axis_idx, :]
            pre = mod_coords[axis_idx, :]
            valid = ~np.isnan(obs) & ~np.isnan(pre)
            if np.any(valid):
                key = ("absolute", model_name, point_MB, axis_name)
                if key not in storage:
                    storage[key] = {"obs": [], "pre": []}
                storage[key]["obs"].append(obs[valid])
                storage[key]["pre"].append(pre[valid])

        # 2. Anatomical coordinate system
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
                        key = ("anatomical", model_name, point_MB, axis_name)
                        if key not in storage:
                            storage[key] = {"obs": [], "pre": []}
                        storage[key]["obs"].append(obs[valid])
                        storage[key]["pre"].append(pre[valid])
            except Exception:
                pass


def build_global_dataframe(storage: dict, dict_frame_to_use: dict) -> pl.DataFrame:
    results_dict = {}

    for (coord_sys, model, point, axis), data in storage.items():
        obs_pooled = np.concatenate(data["obs"])
        pre_pooled = np.concatenate(data["pre"])

        decomp = mae_decomp(obs_pooled, pre_pooled)

        if (coord_sys, model, point) not in results_dict:
            results_dict[(coord_sys, model, point)] = {}
        
        results_dict[(coord_sys, model, point)][axis] = decomp

    dfs = []
    for (coord_sys, model, point), axes_metrics in results_dict.items():
        point_short = point[2:] if point.startswith(("L_", "R_")) else point
        frame_type = dict_frame_to_use.get(point, "unknown") if coord_sys == "anatomical" else "absolute"

        row_data = {
            "coordinate_system": [coord_sys],
            "model": [model],
            "point": [point],
            "point_short": [point_short],
            "frame_type": [frame_type],
        }

        for axis in ["x", "y", "z"]:
            metrics = axes_metrics.get(axis, {"MAE_b": 0.0, "MAE_p": 0.0, "MAE_u": 0.0})
            row_data[f"MAE_b_{axis}"] = [metrics["MAE_b"]]
            row_data[f"MAE_p_{axis}"] = [metrics["MAE_p"]]
            row_data[f"MAE_u_{axis}"] = [metrics["MAE_u"]]

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
    
    storage = {}

    for subject in path_marker_based.iterdir():
        if not subject.is_dir():
            continue
        for task_file in subject.glob("*.c3d"):
            path_mb_file = subject / task_file.name
            path_synth_file = path_synth / subject.name / task_file.name
            path_rtm_file = path_RTM / subject.name / task_file.name

            if path_synth_file.exists() and path_rtm_file.exists():
                print(f"Collecting data from {task_file.name} for subject {subject.name}")
                
                collect_file_data(
                    path_mb_file, path_synth_file,
                    dict_comparaison_point_Synth, dict_frame_to_use,
                    model_name="SynthPose", storage=storage
                )
                
                collect_file_data(
                    path_mb_file, path_rtm_file,
                    dict_comparaison_point_RTM, dict_frame_to_use,
                    model_name="RTMPose", storage=storage
                )

    print("Computing global Robeson regression and MAE decomposition across all pooled subjects and trials...")
    df_errors = build_global_dataframe(storage, dict_frame_to_use)

    output_file = path_data / "Robeson_3Dtrajectories_errors_ALL.parquet"
    df_errors.write_parquet(output_file)
    print(f"Global errors saved to: {output_file}")
    print(df_errors)
    print(f"There are {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")


if __name__ == "__main__":
    main()