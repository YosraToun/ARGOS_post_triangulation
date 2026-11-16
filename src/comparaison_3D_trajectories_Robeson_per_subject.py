from pathlib import Path
import polars as pl
import numpy as np
import ezc3d
import snip_ezc3d as snip
from anatomical_frame import extract_anatomical_frame

#All trials concatenated for each subject

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


def extract_file_data(path_file: Path, dict_comparaison_point: dict):
    """Charge un fichier C3d et retourne les coordonnées 3D brutes pour les points spécifiés."""
    acq = ezc3d.c3d(str(path_file))
    points_data, points_name, _ = snip.get_points_ezc3d(acq)
    
    file_data = {}
    for point_MB, point_model in dict_comparaison_point.items():
        if point_model in points_name and point_MB in points_name:
            idx_model = points_name.index(point_model)
            idx_MB = points_name.index(point_MB)
            
            file_data[point_MB] = {
                "coord_model": points_data[:, idx_model, :],
                "coord_MB": points_data[:, idx_MB, :]
            }
    return file_data


def extract_file_data_anat(path_file: Path, path_marker_based: Path, dict_comparaison_point: dict, dict_frame_to_use: dict):
    """Charge un fichier C3d, extrait le repère anatomique et projette les points."""
    acq = ezc3d.c3d(str(path_file))
    acq_MB = ezc3d.c3d(str(path_marker_based))
    
    points_data, points_name, _ = snip.get_points_ezc3d(acq)
    points_data_MB, points_name_MB, _ = snip.get_points_ezc3d(acq_MB)
    
    file_data_anat = {}
    for point_MB, point_model in dict_comparaison_point.items():
        if point_model in points_name and point_MB in points_name_MB:
            frame_to_use = dict_frame_to_use[point_MB]
            side = point_MB[0]
            
            X_frame, Y_frame, Z_frame = extract_anatomical_frame(points_data_MB, points_name_MB, frame_to_use, side=side)
            
            idx_model = points_name.index(point_model)
            idx_MB = points_name_MB.index(point_MB)
            
            point_proj = project_point_onto_frame(points_data[:, idx_model, :], X_frame, Y_frame, Z_frame)
            point_ref_proj = project_point_onto_frame(points_data_MB[:, idx_MB, :], X_frame, Y_frame, Z_frame)
            
            file_data_anat[point_MB] = {
                "coord_model": point_proj,
                "coord_MB": point_ref_proj
            }
    return file_data_anat


def compute_subject_errors(subject_data_dict):
    """Calcule la décomposition MAE de Robeson sur les listes de coordonnées concaténées pour un sujet."""
    error_results = {}
    for point_MB, coords in subject_data_dict.items():
        ref_concat = np.concatenate([c["coord_MB"] for c in coords], axis=1)
        model_concat = np.concatenate([c["coord_model"] for c in coords], axis=1)

        decomp_x = mae_decomp(ref_concat[0, :], model_concat[0, :])
        decomp_y = mae_decomp(ref_concat[1, :], model_concat[1, :])
        decomp_z = mae_decomp(ref_concat[2, :], model_concat[2, :])

        error_results[point_MB] = {"x": decomp_x, "y": decomp_y, "z": decomp_z}
    return error_results


def error_dict_to_dataframe(
    error_dict: dict,
    subject: str,
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
                "task": ["all_trials_concatenated"],
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
        if not subject.is_dir():
            continue
            
        print(f"Processing subject: {subject.name}")
        
        # Structures pour accumuler les essais par sujet
        subject_synth_abs = {}
        subject_rtm_abs = {}
        subject_synth_anat = {}
        subject_rtm_anat = {}
        
        for task_file in subject.glob("*.c3d"):
            p_synth = path_synth / subject.name / task_file.name
            p_rtm = path_RTM / subject.name / task_file.name
            p_mb = subject / task_file.name
            
            if p_synth.exists() and p_rtm.exists():
                # 1. Extraction Absolue
                file_data_synth = extract_file_data(p_synth, dict_comparaison_point_Synth)
                file_data_rtm = extract_file_data(p_rtm, dict_comparaison_point_RTM)
                
                for pt, data in file_data_synth.items():
                    subject_synth_abs.setdefault(pt, []).append(data)
                for pt, data in file_data_rtm.items():
                    subject_rtm_abs.setdefault(pt, []).append(data)
                
                # 2. Extraction Anatomique
                file_data_anat_synth = extract_file_data_anat(p_synth, p_mb, dict_comparaison_point_Synth, dict_frame_to_use)
                file_data_anat_rtm = extract_file_data_anat(p_rtm, p_mb, dict_comparaison_point_RTM, dict_frame_to_use)
                
                for pt, data in file_data_anat_synth.items():
                    subject_synth_anat.setdefault(pt, []).append(data)
                for pt, data in file_data_anat_rtm.items():
                    subject_rtm_anat.setdefault(pt, []).append(data)

        # Calcul de la décomposition sur les essais concaténés par sujet
        if subject_synth_abs:
            err_synth_abs = compute_subject_errors(subject_synth_abs)
            all_errors.append(error_dict_to_dataframe(err_synth_abs, subject.name, "absolute", dict_frame_to_use, "SynthPose"))
            
        if subject_rtm_abs:
            err_rtm_abs = compute_subject_errors(subject_rtm_abs)
            all_errors.append(error_dict_to_dataframe(err_rtm_abs, subject.name, "absolute", dict_frame_to_use, "RTMPose"))
            
        if subject_synth_anat:
            err_synth_anat = compute_subject_errors(subject_synth_anat)
            all_errors.append(error_dict_to_dataframe(err_synth_anat, subject.name, "anatomical", dict_frame_to_use, "SynthPose"))
            
        if subject_rtm_anat:
            err_rtm_anat = compute_subject_errors(subject_rtm_anat)
            all_errors.append(error_dict_to_dataframe(err_rtm_anat, subject.name, "anatomical", dict_frame_to_use, "RTMPose"))

    if all_errors:
        df_errors = pl.concat(all_errors)
        output_file = path_data / "Robeson_3Dtrajectories_errors_per_subject.parquet"
        df_errors.write_parquet(output_file)
        print(f"Errors saved to: {output_file}")
        print(df_errors)
        print(f"There are {df_errors.shape[0]} rows and {df_errors.shape[1]} columns in the dataframe")
    else:
        print("No valid data found to process.")


if __name__ == "__main__":
    main()