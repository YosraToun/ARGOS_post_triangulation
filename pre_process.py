import ezc3d
from pathlib import Path
import numpy as np
from signal_processing import (
    transforms_zero_to_nan,
    extract_trim_from_marker_based,
    resample_point_data,
    filter_point_data_with_nan_segments,
    XYZ_to_ZXY,
)
from anatomical_regression import calculate_RAB, calculate_mid_point
from c3d_io import load_c3d, write_new_c3d
import snip_ezc3d as snip


def correct_markerless_fq_folder(
    folder_marker_based: Path, folder_markerless_to_correct: Path, folder_marker_less_to_export: Path
):
    """
    Correct the frame rate of marker-less data based on the corresponding marker-based data.
    """

    for subject_folder in folder_marker_based.iterdir():
        if not subject_folder.is_dir():
            continue
        subject_name = subject_folder.name
        print(f"\n--- Processing subject: {subject_name} ---")

        folder_subject_markerbase = folder_marker_based / subject_name / "labelled_data"

        # Check if the subject folder exists in the markerless folder
        subject_name_ml = "Sujet_0" + subject_name[-2:]
        ml_subject_folder = folder_markerless_to_correct / subject_name_ml
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue

        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_marker_less_to_export / subject_name
        output_subject_folder.mkdir(parents=True, exist_ok=True)

        for trial in folder_subject_markerbase.glob("*.c3d"):
            trial_name = trial.stem
            print(f"Processing trial: {trial_name}")
            # Check if the corresponding markerless trial exists
            ml_trial_path = ml_subject_folder / f"{trial_name}.c3d"
            if not ml_trial_path.exists():
                print(f"❌ No corresponding markerless trial for {trial_name}")
                continue
            correct_markerless_fq_file(trial, ml_trial_path, output_subject_folder)


def correct_markerless_fq_file(trial: Path, ml_trial_path: Path, output_subject_folder: Path):
    trial_name = trial.stem
    # Read the marker-based and markerless c3d files
    c3d_mb, points_mb, fq_mb, labels_mb = load_c3d(trial)
    c3d_ml, points_ml, fq_ml, labels_ml = load_c3d(ml_trial_path)

    # try:
    #         c3d_ml, points_ml, fq_ml, labels_ml = load_c3d(ml_trial_path)
    # except Exception as e:
    #         print(f"❌ Skipping corrupted or unreadable file {ml_trial_path.name}: {e}")
    #         return


    # get the frame count of the two c3d files
    nb_frames_mb = points_mb.shape[2]
    nb_frames_ml = points_ml.shape[2]

    # calculate the ratio of the two frame counts
    ratio = nb_frames_mb / nb_frames_ml
    # find if the ratio is close to 1 or 2
    ratio_rounded = round(ratio)

    if not np.isclose(ratio, ratio_rounded, atol=0.1):
        print(f"❌ The ratio of the frame counts is not close to 1 or 2 for {trial_name}. Ratio: {ratio}")
        return

    # get fq of mb
    if ratio_rounded == 2:
        c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
        # check if the rotation rate is present in the marker-based c3d file
        if "ROTATION" in c3d_mb["parameters"]:
            c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"] / 2
    elif ratio_rounded == 1:
        c3d_ml["parameters"]["POINT"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
        # check if the rotation rate is present in the marker-based c3d file
        if "ROTATION" in c3d_mb["parameters"]:
            c3d_ml["parameters"]["ROTATION"]["RATE"]["value"] = c3d_mb["parameters"]["POINT"]["RATE"]["value"]
    else:
        print(f"❌ The ratio of the frame counts is not 1 or 2 for {trial_name}. Ratio: {ratio}")
        # stop here if the ratio is not 1 or 2
        return

    print(str(output_subject_folder / f"{trial_name}.c3d"))
    c3d_ml.write(str(output_subject_folder / f"{trial_name}.c3d"))


def pre_processed_marker_based_folder(marker_based_folder: Path, output_folder: Path):

    for subject_dir in sorted(marker_based_folder.iterdir()):
        if subject_dir.is_dir() and subject_dir.name.startswith("S") and subject_dir.name[1:].isdigit():
            subject_name = subject_dir.name  # Extracts subject ID (e.g., "S09")

            header_text = (
                f"=================================================\n"
                f"      Processing Subject: {subject_name}\n"
                f"=================================================\n"
            )
            print(header_text)

            raw_folder = subject_dir / "labelled_data"
            preprocess_output_folder = output_folder / subject_name

            # Verify that the 'labelled_data' folder exists before proceeding
            if not raw_folder.exists():
                print(f"Warning: 'labelled_data' folder not found for {subject_name}. Skipping.\n")
                continue

            # Automatically ensure 'Preprocess' folder exists
            preprocess_output_folder.mkdir(parents=True, exist_ok=True)

            # Find all C3D files in this subject's 'labelled_data' folder
            c3d_files = list(raw_folder.glob("*.c3d"))
            total_files = len(c3d_files)
            print(f"Found {total_files} C3D files for {subject_name}\n")

            # =========================================================================
            #      D. TRIMMING, NaN VERIFICATION & UPPER LIMBS JOINT DETERMINATION
            # =========================================================================
            for file_path in c3d_files:
                pre_processed_marker_based_files(file_path=file_path, output_folder=preprocess_output_folder)


def pre_processed_marker_based_files(file_path: Path, output_folder: Path):
    file_name = file_path.name
    target_fps = 60.0  # Target frame rate for resampling
    print(f"Processing: {file_name}")

    c3d, points, fq, labels = load_c3d(file_path)

    # 2. Trim and log
    to_trim, start_idx, end_idx = extract_trim_from_marker_based(file_path)
    trim_points = points[:, :, start_idx:end_idx]

    # filter and resample
    filtered_points = filter_point_data_with_nan_segments(points=trim_points, fs=fq, cutoff=6.0, order=2, max_gap=10)
    resampled_points = resample_point_data(filtered_points, original_frame_rate=fq, target_frame_rate=target_fps)

    # Calculate GH with RAB to add in the data
    R_GH, L_GH = calculate_RAB(resampled_points, labels)
    # R_FJC and L_FJC correspond to the middle of the HM2 and HM5
    R_EJC, L_EJC, R_WJC, L_WJC, R_FJC, L_FJC = calculate_mid_point(resampled_points, labels)

    # Calculate joint centers and add them to the resampled_points

    exported_c3d_path = output_folder / f"{file_path.stem}.c3d"
    name_points = c3d["parameters"]["POINT"]["LABELS"]["value"]
    # Add the R_GH and L_GH to the resampled_points
    resampled_points = np.concatenate((resampled_points, R_GH[:, np.newaxis, :], L_GH[:, np.newaxis, :]), axis=1)
    resampled_points = np.concatenate(
        (
            resampled_points,
            R_EJC[:, np.newaxis, :],
            L_EJC[:, np.newaxis, :],
            R_WJC[:, np.newaxis, :],
            L_WJC[:, np.newaxis, :],
            R_FJC[:, np.newaxis, :],
            L_FJC[:, np.newaxis, :],
        ),
        axis=1,
    )
    name_points.append("R_GH")
    name_points.append("L_GH")
    name_points.append("R_EJC")
    name_points.append("L_EJC")
    name_points.append("R_WJC")
    name_points.append("L_WJC")
    name_points.append("R_FJC")
    name_points.append("L_FJC")

    write_new_c3d(resampled_points, name_points, target_fps, str(output_folder / f"{file_name}"))


def pre_processed_marker_less_folder(
    folder_marker_based: Path,
    folder_markerless_to_correct: Path,
    folder_marker_less_to_export: Path,
    remove_nan: bool = True,
    filter_data: bool = True,
    cutoff: float = 6.0,
    order: int = 4,
):
    """
    Pre-process a folder of markerless data based on the corresponding marker-based data.
    Parameters:
    -----------
    folder_marker_based : Path
        Path to the folder containing the marker-based data.
    folder_markerless_to_correct : Path
        Path to the folder containing the markerless data to be corrected.
    folder_marker_less_to_export : Path
        Path to the folder where the corrected markerless data will be exported.
    remove_nan : bool, optional
        Whether to remove NaN values from the data, by default True
    filter_data : bool, optional
        Whether to filter the data, by default True
    cutoff : float, optional
        Cutoff frequency for the filter, by default 6.0
    order : int, optional
        Order of the filter, by default 4
    """
    for subject_folder in folder_marker_based.iterdir():
        if not subject_folder.is_dir():
            continue
        subject_name = subject_folder.name
        print(f"\n--- Processing subject: {subject_name} ---")
        folder_subject_markerbase = folder_marker_based / subject_name / "labelled_data"

        # Check if the subject folder exists in the markerless folder
        ml_subject_folder = folder_markerless_to_correct / subject_name
        if not ml_subject_folder.exists():
            print(f"❌ No markerless data for {subject_name}")
            continue

        # Create the output folder for the corrected markerless data
        output_subject_folder = folder_marker_less_to_export / subject_name
        output_subject_folder.mkdir(parents=True, exist_ok=True)

        for trial in folder_subject_markerbase.glob("*.c3d"):
            trial_name = trial.stem
            print(f"Processing trial: {trial_name}")
            # Check if the corresponding markerless trial exists
            ml_trial_path = ml_subject_folder / f"{trial_name}.c3d"
            mb_trial_path = folder_subject_markerbase / f"{trial_name}.c3d"
            if not ml_trial_path.exists():
                print(f"❌ No corresponding markerless trial for {trial_name}")
                continue
            pre_processed_marker_less_files(
                ml_trial_path,
                mb_trial_path,
                output_subject_folder,
                remove_nan=remove_nan,
                filter_data=filter_data,
                cutoff=cutoff,
                order=order,
            )


def pre_processed_marker_less_files(
    ml_trial_path: Path,
    mb_trial_path: Path,
    output_subject_folder: Path,
    remove_nan: bool = True,
    filter_data: bool = True,
    cutoff: float = 6.0,
    order: int = 4,
):
    """
    Pre-processing of a single file of markerless data based on the corresponding marker-based data.
    """
    trial_name = ml_trial_path.stem
    print(f"Processing trial: {trial_name}")
    # Check if the corresponding markerless trial exists

    # remove NaN from the markerless trial
    c3d_ml, points_ml, fq_ml, name_points = load_c3d(ml_trial_path)
    c3d_mb, points_mb, fq_mb, name_points_mb = load_c3d(mb_trial_path)

    point_ml_reoriented = XYZ_to_ZXY(points_ml)
    # Remove [0,0,0] in markers and replace them with NaN
    if remove_nan:
        points_ml_NaN = transforms_zero_to_nan(point_ml_reoriented)
    else:
        points_ml_NaN = point_ml_reoriented

    if filter_data:
        points_ml_filtered = filter_point_data_with_nan_segments(
            points_ml_NaN, fq_ml, cutoff=cutoff, order=order, max_gap=10
        )
    else:
        points_ml_filtered = points_ml_NaN

    # extract from the markerbased trial the cut of the markerless trial
    to_trim, start_idx, end_idx = extract_trim_from_marker_based(mb_trial_path)
    # calculate the ratio of the frame rates of the two c3d files
    ratio = fq_mb / fq_ml
    ratio_rounded = round(ratio)
    start_idx = int(start_idx / ratio_rounded)
    end_idx = int(end_idx / ratio_rounded)
    points_ml_trimmed = points_ml_filtered[:, :, start_idx:end_idx]

    write_new_c3d(points_ml_trimmed, name_points, fq_ml, str(output_subject_folder / f"{trial_name}.c3d"))


def fusion_markerless_model(path_synthpose: Path, path_rtmpose: Path, path_export: Path):
    """
    Fusion of the markerless data from different models into one file.
    """

    list_points_RTMPOSE = [
        "L_elbow",
        "L_base_hand",
        "L_MCP_thumb",
        "L_MCP_index",
        "L_MCP_middle",
        "L_MCP_ring",
        "L_MCP_little",
        "R_elbow",
        "R_base_hand",
        "R_MCP_thumb",
        "R_MCP_index",
        "R_MCP_middle",
        "R_MCP_ring",
        "R_MCP_little",
    ]

    # extract all_subject in path_synthpose
    all_subjects_synthpose = [f for f in path_synthpose.iterdir() if f.is_dir()]
    print(f"Found {len(all_subjects_synthpose)} subjects in {path_synthpose}")
    print(f"Found {all_subjects_synthpose}")
    # for each subject, extract all trials in path_synthpose
    for subject in all_subjects_synthpose:
        print(f"Processing subject: {subject.name}")
        # extract all c3d files in the subject folder
        all_trials_synthpose = [f for f in subject.iterdir() if f.is_file() and f.suffix == ".c3d"]
        name_subject = subject.name
        if len(all_trials_synthpose) == 0:
            print(f"No c3d file found in {subject}")
            continue

        # Check if the folder of the subject exist in path_export, if not create it
        if not (path_export / name_subject).exists():
            (path_export / name_subject).mkdir(parents=True, exist_ok=True)

        for trial in all_trials_synthpose:
            print(f"Processing trial: {trial}")
            trial_name = trial.stem
            print(f"Trial name: {trial_name}")
            # read the c3d file
            c3d_synthpose = ezc3d.c3d(str(trial))
            c3d_rtmpose = ezc3d.c3d(str(path_rtmpose / name_subject / f"{trial_name}.c3d"))
            # create a dictionary with the points to add from c3d_rtmpose
            points_to_add = dict()

            for point_name in list_points_RTMPOSE:
                if point_name in c3d_rtmpose["parameters"]["POINT"]["LABELS"]["value"]:
                    point_index = c3d_rtmpose["parameters"]["POINT"]["LABELS"]["value"].index(point_name)
                    point_data = c3d_rtmpose["data"]["points"][:3, point_index, :]

                    points_to_add[point_name] = point_data
                else:
                    print(f"Point {point_name} not found in {trial_name} of {name_subject}")

            acq = snip.add_point_from_dictionary(c3d_synthpose, points_to_add)
            # export the new c3d file in path_export
            acq.write(str(path_export / name_subject / f"{trial_name}.c3d"))


def fusion_model(path_synthpose: Path, path_rtmpose: list, path_export: Path):
    """
    Fusion of the markerless data from different models into one file.
    """
    # extract all_subject in path_synthpose
    all_subjects_synthpose = [f for f in path_synthpose.iterdir() if f.is_dir()]
    print(f"Found {len(all_subjects_synthpose)} subjects in {path_synthpose}")
    print(f"Found {all_subjects_synthpose}")
    # for each subject, extract all trials in path_synthpose
    for subject in all_subjects_synthpose:
        print(f"Processing subject: {subject.name}")
        # extract all c3d files in the subject folder
        all_trials_synthpose = [f for f in subject.iterdir() if f.is_file() and f.suffix == ".c3d"]
        name_subject = subject.name
        if len(all_trials_synthpose) == 0:
            print(f"No c3d file found in {subject}")
            continue

        # Check if the folder of the subject exist in path_export, if not create it
        if not (path_export / name_subject).exists():
            (path_export / name_subject).mkdir(parents=True, exist_ok=True)

        for trial in all_trials_synthpose:
            print(f"Processing trial: {trial}")
            trial_name = trial.stem
            print(f"Trial name: {trial_name}")
            # read the c3d file
            c3d_synthpose = ezc3d.c3d(str(trial))
            c3d_rtmpose = ezc3d.c3d(str(path_rtmpose / name_subject / f"{trial_name}.c3d"))
            # create a dictionary with the points to add from c3d_rtmpose
            points_to_add = dict()
            list_points_name_to_add = c3d_rtmpose["parameters"]["POINT"]["LABELS"]["value"]

            for point_name in list_points_name_to_add:
                if point_name in c3d_rtmpose["parameters"]["POINT"]["LABELS"]["value"]:
                    point_index = c3d_rtmpose["parameters"]["POINT"]["LABELS"]["value"].index(point_name)
                    point_data = c3d_rtmpose["data"]["points"][:3, point_index, :]

                    points_to_add[point_name] = point_data
                else:
                    print(f"Point {point_name} not found in {trial_name} of {name_subject}")

            acq = snip.add_point_from_dictionary(c3d_synthpose, points_to_add)
            # export the new c3d file in path_export
            acq.write(str(path_export / name_subject / f"{trial_name}.c3d"))
