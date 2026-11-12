import snip_ezc3d as snip
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import ezc3d 
# fonction 1 : Generate the homogeneoux matrix of the arm from the marker-based data
import polars as pl
import pandas as pd

dict_task = {
    "00-static-stand" : "static",
    "01-eat-yaourt" : "eat",
    "02-cut-food" : "cut",
    "03-open-bottle" : "open bottle",
    "03-Open_a_Bottle_and_Pour" : "open and pour",
    "04-Hand_to_Head" : "hand to head",
    "04-take-in-box" : "take in box",
    "05-use-computer" : "computer",
    "06-drawing": "drawing",
    "06-Open_Kit" : "open kit",
    "07-use-ruler" : "ruler",
    "08-cut-scissor": "cut scissor",
    "09-erase-eraser": "erase eraser",
    "10-reach_and_grasp_far" : "R&G far",
    "11-reach_and_grasp_mid" : "R&G mid",
    "12-reach_and_grasp_near" : "R&G near",
    "13-playdoe" : "playdoe",
    "14-put-toothpaste" : "put toothpaste",
    "15-brush-teeth" : "brush teeth",
    "16-comb-hair" : "comb hair",
    "17-hand-to-back" : "hand to back",
    "18-lift-tray" : "lift tray",
    "20-catch-throw-ball" : "catch throw ball",
    "21-abduction" : "GH abduction",
    "22-flexion" : "GH flexion",
    "23-rotation-plus-supination-full-arm-extended-front" : "GH rot + El sup extend",
    "24-Elbow_Flexion_Extension" : "Elbow flexion",
    "24-Elbow_Pronosupination_Bras_Bend" : "Elbow pronosupination",
    "26-Cymbals": "cymbals",
}
def define_frame_arm(points_data,points_name,elbow_flexion_axis,GH_name,EJC_name):
    # Get the indices of the markers
    GH_index = points_name.index(GH_name)
    EJC_index = points_name.index(EJC_name)

    # Get the positions of the markers
    GH = points_data[:, GH_index, :]
    EJC = points_data[:, EJC_index, :]

    # Calculate the origin of the arm frame (EJC)
    origin = EJC

    # Calculate the x-axis (from EJC to WJC)
    y_axis = GH - EJC
    y_axis /= np.linalg.norm(y_axis, axis=0)

    # Calculate the z-axis (normal to the plane formed by GH, EJC, and WJC)
    z_axis = elbow_flexion_axis

    # Calculate the x-axis (cross product of z and y axes)
    x_axis = np.cross(z_axis.T, y_axis.T).T
    x_axis /= np.linalg.norm(x_axis, axis=0)

    # renormalize the y-axis to ensure orthogonality
    y_axis = np.cross(z_axis.T, x_axis.T).T
    y_axis /= np.linalg.norm(y_axis, axis=0)

    # Create the homogeneous transformation matrix for each frame
    n_frames = points_data.shape[2]
    H_arm = np.zeros((4, 4, n_frames))
    
    for i in range(n_frames):
        H_arm[0:3, 0, i] = x_axis[:, i]
        H_arm[0:3, 1, i] = y_axis[:, i]
        H_arm[0:3, 2, i] = z_axis[:, i]
        H_arm[0:3, 3, i] = origin[:, i]
        H_arm[3, :, i] = [0, 0, 0, 1]

    return H_arm

def compute_norm_to_plane_arm(c3d_path, GH_name, EJC_name, WJC_name):

    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)

    GH = points_data[:, points_name.index(GH_name), :]
    EJC = points_data[:, points_name.index(EJC_name), :]
    WJC = points_data[:, points_name.index(WJC_name), :]

    # vectors defining the arm plane
    v1 = WJC - EJC
    v2 = GH - EJC

    # --------------------------------------------------
    # angle between v1 and v2
    # --------------------------------------------------

    v1_norm = v1 / np.linalg.norm(v1, axis=0)
    v2_norm = v2 / np.linalg.norm(v2, axis=0)

    dot = np.sum(v1_norm * v2_norm, axis=0)
    dot = np.clip(dot, -1.0, 1.0)

    angle_v1_v2 = np.rad2deg(np.arccos(dot))

    # --------------------------------------------------
    # normal to plane
    # --------------------------------------------------

    normal = np.cross(v1.T, v2.T)

    normal_norm = np.linalg.norm(normal, axis=1)

    valid = normal_norm > 1e-10

    normal[valid] /= normal_norm[valid][:, np.newaxis]
    normal[~valid] = np.nan

    normal = normal.T

    frame = define_frame_arm(points_data, points_name, normal, GH_name=GH_name, EJC_name=EJC_name)

    return frame, angle_v1_v2




def compute_elbow_flexion_axis(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the elbow flexion axis
    EL = points_data[:, points_name.index("R_EL"), :]
    EM = points_data[:, points_name.index("R_EM"), :]
    # Calculate the elbow flexion axis
    elbow_flexion_axis = EL - EM
    elbow_flexion_axis /= np.linalg.norm(elbow_flexion_axis, axis=0)

    frame = define_frame_arm(points_data,points_name,elbow_flexion_axis, GH_name="R_GH", EJC_name="R_EJC")

    return frame

def compute_elbow_flexion_axis_synthpose(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the elbow flexion axis
    EL = points_data[:, points_name.index("r_lelbow"), :]
    EM = points_data[:, points_name.index("r_melbow"), :]
    # Calculate the elbow flexion axis
    elbow_flexion_axis = EL - EM
    elbow_flexion_axis /= np.linalg.norm(elbow_flexion_axis, axis=0)

    frame = define_frame_arm(points_data,points_name,elbow_flexion_axis, GH_name="R_Shoulder", EJC_name="R_Elbow")

    return frame

def compute_elbow_flexion_axis_fakeRtmpose(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the elbow flexion axis
    EL = points_data[:, points_name.index("r_lelbow"), :]
    EM = points_data[:, points_name.index("r_melbow"), :]
    # Calculate the elbow flexion axis
    elbow_flexion_axis = EL - EM
    elbow_flexion_axis /= np.linalg.norm(elbow_flexion_axis, axis=0)

    frame = define_frame_arm(points_data,points_name,elbow_flexion_axis, GH_name="R_Shoulder", EJC_name="R_Elbow")

    return frame

def compare_frames(H_plane, H_isb):

    n_frames = H_plane.shape[2]

    quaternions = np.zeros((n_frames, 4))
    angles = np.zeros(n_frames)

    for i in range(n_frames):

        R_plane = H_plane[:3, :3, i]
        R_isb   = H_isb[:3, :3, i]
        if np.isnan(R_plane).any() or np.isnan(R_isb).any():
            quaternions[i, :] = np.array([np.nan, np.nan, np.nan, np.nan])
            angles[i] = np.nan
            continue
        R_rel = R_plane @ R_isb.T

        rot = Rot.from_matrix(R_rel)

        quaternions[i, :] = rot.as_quat()  # [x,y,z,w]

        angles[i] = np.rad2deg(rot.magnitude())

    return quaternions, angles

def comp_method(c3d_path):
    frame_MB_normal_plane, angle_v1_v2 = compute_norm_to_plane_arm(c3d_path, GH_name="R_GH", EJC_name="R_EJC", WJC_name="R_WJC")
    frame_ISB = compute_elbow_flexion_axis(c3d_path)
    frame_synthpose = compute_elbow_flexion_axis_synthpose(c3d_path)
    frame_ML_normal_plane,angle = compute_norm_to_plane_arm(c3d_path, GH_name="R_Shoulder", EJC_name="R_Elbow", WJC_name="R_Wrist")
    # comparaison of the two methods
    quaternion_mb,error_angles_normal_plane_model = compare_frames(frame_MB_normal_plane, frame_ISB)
    quaternion_ml,error_angles_synthpose = compare_frames(frame_synthpose, frame_ISB)
    quaternion_fakeRtm,error_angles_ML_normal_plane = compare_frames(frame_ML_normal_plane, frame_ISB)
    quaternion,error_angles_MLvsMB_normal_plane = compare_frames(frame_MB_normal_plane, frame_ML_normal_plane)





    return error_angles_normal_plane_model,error_angles_synthpose,error_angles_ML_normal_plane,error_angles_MLvsMB_normal_plane,angle_v1_v2

def batch_process(reference_root,filename_output):
    """Process a batch of c3d files and compare the two methods for elbow flexion axis estimation and generate a parquet file for all subject and task with the angles between the two methods (in degrees)
    Parameters
    ----------
    reference_root : _type_
        _description_
    """
    rows = []
    for subject_dir in sorted(reference_root.glob("S*")):
        subject = subject_dir.name
        print(f"Processing {subject}")
        for c3d_file in sorted(subject_dir.glob("*.c3d")):
            task = c3d_file.stem
            print(f"   {task}")
            error_angles_normal_plane_model, error_angles_synthpose, error_angles_ML_normal_plane, error_angles_MLvsMB_normal_plane, angle_v1_v2 = comp_method(c3d_file)
            for angle, plane_angle in zip(error_angles_normal_plane_model, angle_v1_v2):
                rows.append({"subject": subject,"Elbow flexion axis":"normal plane MB", "task": dict_task[task], "angle": angle, "plane_angle": plane_angle})
            for angle,plane_angle in zip(error_angles_synthpose, angle_v1_v2):
                rows.append({"subject": subject,"Elbow flexion axis":"Synthpose", "task": dict_task[task], "angle": angle, "plane_angle": plane_angle})
            for angle,plane_angle in zip(error_angles_ML_normal_plane, angle_v1_v2):
                rows.append({"subject": subject,"Elbow flexion axis":"normal plane ML", "task": dict_task[task], "angle": angle, "plane_angle": plane_angle})
            for angle,plane_angle in zip(error_angles_MLvsMB_normal_plane, angle_v1_v2):
                rows.append({"subject": subject,"Elbow flexion axis":"normal plane ML vs MB", "task": dict_task[task], "angle": angle, "plane_angle": plane_angle})
    df = pl.DataFrame(rows)

    print(df.shape)
    # check if the folder exists, if not create it
    filename_output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        filename_output
    )   


def generate_stats_from_parquet(filename_input, filename_output):
    """
    Generate an Excel file with:

    Sheet 1: statistics for each subject and task
    Sheet 2: statistics for each task across all subjects

    Statistics:
    - mean
    - std
    - median
    - min
    - max
    - n_frames
    """
    df = pl.read_parquet(filename_input)

    # Remove NaNs if present
    df = df.filter(~pl.col("angle").is_nan())

    # -------------------------------------------------------------
    # Subject × Task
    # -------------------------------------------------------------
    stats_subject_task = (
        df.group_by(["subject", "task","Elbow flexion axis"])
        .agg(
            [
                pl.len().alias("n_frames"),
                pl.col("angle").mean().alias("mean_deg"),
                pl.col("angle").std().alias("std_deg"),
                pl.col("angle").median().alias("median_deg"),
                pl.col("angle").min().alias("min_deg"),
                pl.col("angle").max().alias("max_deg"),
            ]
        )
        .sort(["subject", "task"])
    )

    # -------------------------------------------------------------
    # Task across all subjects
    # -------------------------------------------------------------
    stats_task = (
        df.group_by(["task", "Elbow flexion axis"])
        .agg(
            [
                pl.len().alias("n_frames"),
                pl.col("angle").mean().alias("mean_deg"),
                pl.col("angle").std().alias("std_deg"),
                pl.col("angle").median().alias("median_deg"),
                pl.col("angle").min().alias("min_deg"),
                pl.col("angle").max().alias("max_deg"),
            ]
        )
        .sort("task")
    )

    # -------------------------------------------------------------
    # Export to Excel
    # -------------------------------------------------------------
    with pd.ExcelWriter(filename_output) as writer:

        stats_subject_task.to_pandas().to_excel(
            writer,
            sheet_name="subject_task",
            index=False,
        )

        stats_task.to_pandas().to_excel(
            writer,
            sheet_name="task_global",
            index=False,
        )

    print(f"Saved statistics to: {filename_output}")

    return stats_subject_task, stats_task

def generate_boxplots_from_parquet(filename_input, output_dir):
    """
    Generate:
    - one boxplot per subject (all tasks)
    - one global boxplot (all subjects)

    Parameters
    ----------
    filename_input : Path
        parquet file generated by batch_process
    output_dir : Path
        output directory for figures
    """

    import polars as pl
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(filename_input)

    df = (
        df.filter(~pl.col("angle").is_nan())
          .to_pandas()
    )

    sns.set_style("whitegrid")

    # --------------------------------------------------
    # One figure per subject
    # --------------------------------------------------

    subjects = sorted(df["subject"].unique())

    for subject in subjects:

        df_sub = df[df["subject"] == subject]

        plt.figure(figsize=(12, 6))

        sns.boxplot(
            data=df_sub,
            x="task",
            y="angle",
            hue="Elbow flexion axis",
        )

        plt.title(f"{subject} - Angle difference between methods")
        plt.xlabel("Task")
        plt.ylabel("Angle (deg)")
        plt.xticks(rotation=90)
        plt.legend(title="Method")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{subject}_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------
    # Global figure
    # --------------------------------------------------
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="task",
        y="angle",
        hue="Elbow flexion axis",
    )

    plt.title("All subjects - Angle difference between methods")
    plt.xlabel("Task")
    plt.ylabel("Angle (deg)")
    plt.xticks(rotation=90)
    plt.legend(title="Method", loc="upper right")
    plt.tight_layout()

    plt.savefig(
        output_dir / "all_subjects_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Error vs plane angle
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="plane_angle",
        y="angle",
        hue="Elbow flexion axis",
        alpha=0.05,
        s=3
    )

    plt.xlabel("Elbow plane angle (deg)")
    plt.ylabel("Quaternion angle difference with the reference (deg)")
    plt.title("Error as a function of arm plane opening angle")

    plt.tight_layout()

    plt.savefig(
        output_dir / "error_vs_plane_angle.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    df["plane_bin"] = pd.cut(
    df["plane_angle"],
    bins=np.arange(0, 185, 5))
    plt.figure(figsize=(12,6))

    sns.boxplot(
        data=df,
        x="plane_bin",
        y="angle",
        hue="Elbow flexion axis",
        showfliers=False
    )

    plt.xticks(rotation=90)
    plt.xlabel("Plane opening angle at the elbow (5° bins)")
    plt.ylabel("Difference between methods (deg)")
    plt.title("Error vs arm plane angle")

    plt.tight_layout()

    plt.savefig(
        output_dir / "error_vs_plane_angle_boxplot.png",
        dpi=300,
    )
    plt.close()
    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="Elbow flexion axis",
        y="angle"
    )

    plt.ylabel("Difference to ISB frame (deg)")
    plt.xlabel("")
    plt.title("Comparison of arm frame definition methods")

    plt.tight_layout()
    plt.savefig(
        output_dir / "model_comparison.png",
        dpi=300
    )
    plt.close()

    print(f"Figures saved in {output_dir}")


if __name__ == "__main__":
    # Example usage
    # c3d_file_path = "data/pre_processed/marker_based/S14/05-use-computer.c3d"
    # print("Comparing the two methods for elbow flexion axis estimation...")
    # angles = comp_method(c3d_file_path)
    # print("Angles between the two methods (in degrees):")
    # print(angles)
    reference_root = Path("data/pre_processed/marker_less/SynthRTMposeMB_Marker")
    filename_parquet = Path("figures") / "angles_between_methods_elbow" / "angles_between_methods_elbow.parquet"
    filename_excel = Path("figures") / "angles_between_methods_elbow" / "stats_angles_between_methods_elbow.xlsx"
    figure_folder = Path("figures") / "angles_between_methods_elbow"


    print(f"Processing batch of c3d files and saving to {filename_parquet}...")
    batch_process(reference_root, filename_parquet)
    generate_stats_from_parquet(filename_parquet, filename_excel)
    generate_boxplots_from_parquet(filename_parquet, figure_folder)

