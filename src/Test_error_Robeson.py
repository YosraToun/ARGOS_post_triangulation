# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:44:48 2026

@author: ytounekt
"""

# -*- coding: utf-8 -*- 
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from ezc3d import c3d

def mae_decomp(obs, pre):
    """
    Decomposes Mean Absolute Error (MAE) into bias (MAE_b), 
    proportionality (MAE_p), and unsystematic (MAE_u) components 
    based on Robeson & Willmott (2023).
    """
    obs = np.asarray(obs, dtype=float)
    pre = np.asarray(pre, dtype=float)
    n = len(obs)
    
    # 1. Absolute errors and Mean Bias Error (MBE)
    abserr = np.abs(pre - obs)
    mbe = np.mean(pre) - np.mean(obs)
    pre_prime = pre - mbe
    
    # 2. OLS regression of pre_prime on obs (slope and intercept)
    slope, intercept = np.polyfit(obs, pre_prime, 1)
    phat_prime = intercept + slope * obs
    
    # 3. Form absolute error weights
    b = np.abs(mbe)
    p = np.abs(phat_prime - obs)
    u = np.abs(pre_prime - phat_prime)
    
    denom = b + p + u
    
    # Handle any potential division by zero for exact match points
    with np.errstate(divide='ignore', invalid='ignore'):
        bias = np.where(denom == 0, 0, b / denom)
        prop = np.where(denom == 0, 0, p / denom)
        unsys = np.where(denom == 0, 0, u / denom)
        
    # 4. Calculate final decomposed components
    maeb = np.sum(bias * abserr) / n
    maep = np.sum(prop * abserr) / n
    maeu = np.sum(unsys * abserr) / n
    
    total_mae = np.sum(abserr) / n
    
    return {
        "MAE_b": maeb,
        "MAE_p": maep,
        "MAE_u": maeu,
        "Total_MAE": total_mae,
        "Sum_Check": maeb + maep + maeu
    }

def load_marker_data(file_path, marker_name, axis=2):
    """
    Loads a specific marker's time-series coordinate from a C3D file.
    
    Parameters:
    -----------
    file_path : str
        Path to the .c3d file.
    marker_name : str
        Name of the marker (e.g., 'R_Knee', 'LASI').
    axis : int
        0 for X, 1 for Y, 2 for Z coordinate.
    """
    data = c3d(file_path)
    
    # Retrieve marker labels from parameters
    labels = list(data['parameters']['POINT']['LABELS']['value'])
    
    if marker_name not in labels:
        raise ValueError(f"Marker '{marker_name}' not found in {file_path}. Available markers: {labels}")
    
    marker_idx = labels.index(marker_name)
    
    # Extract coordinate across all frames: shape (4, n_points, n_frames)
    trajectory = data['data']['points'][axis, marker_idx, :]
    return trajectory

# --- Example Usage with C3D files ---
if __name__ == "__main__":
    # Define your file paths and target marker
    obs_c3d_path = r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\workspace\ARGOS_post_triangulation\data\pre_processed\marker_less\SynthRTMPose\S08\00-static-stand.c3d"
    pred_c3d_path = r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\workspace\ARGOS_post_triangulation\data\pre_processed\marker_based\S08\00-static-stand.c3d"
    target_marker_mb = "R_GH"  # Replace with your marker label
    target_marker_ml = "R_Shoulder"  # Replace with your marker label
    target_axis = 1       # 2 for Z axis (height/vertical), 1 for Y, 0 for X
    
    try:
        # Load trajectories from files
        observations = load_marker_data(obs_c3d_path, target_marker_ml, axis=target_axis)
        predictions = load_marker_data(pred_c3d_path, target_marker_mb, axis=target_axis)
        
        # Run decomposition
        results = mae_decomp(observations, predictions)
        print(f"--- MAE Decomposition Results for marker '{target_marker_ml}' ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
    except FileNotFoundError:
        print("Please update the file paths to point to valid C3D files on your system.")
    except Exception as e:
        print(f"An error occurred: {e}")