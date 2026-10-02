from pathlib import Path
from pre_process import (
    correct_markerless_fq_folder,
    pre_processed_marker_based_folder,
    pre_processed_marker_less_folder,
    fusion_markerless_model,
    fusion_model,
)

# defintion of the folder of the data
data_folder_raw_marker_based = Path(r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\ARGOS_data\data\raw\marker_based")
data_folder_raw_marker_less = Path(r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\ARGOS_data\data\raw\marker_less\triangulation_20px")


# pre_processed data folder
pre_processed_data_folder = Path(r"D:\Yosra\Engineering\Travail_LBMC\Projet_ARGOS\ARGOS_data\data\pre_processed")
pp_marker_based_folder = pre_processed_data_folder / "marker_based"
pp_marker_less_folder = pre_processed_data_folder / "marker_less"

# file containing the cut of the marker-based data


def pre_process_marker_based_data():
    """
    Pre-process the marker-based data and save it to the pre-processed folder.
    """
    # Trim the marker-based data based on the file
    pre_processed_marker_based_folder(data_folder_raw_marker_based, pp_marker_based_folder)


def pre_process_marker_less_data():
    """
    Pre-process the marker-less data and save it to the pre-processed folder.
    """
    # for each marker-less model, pre-process the data and save it to the pre-processed folder.
    # take into account that it is possible that there is only one marker-less model, but the code should be able to handle multiple models.
    marker_less_model = ["SynthPose", "RTMPose"]
    for model in marker_less_model:
        # Correction of the frame rate of the marker-less data based on the corresponding marker-based data.
        folder_markerless_to_correct = data_folder_raw_marker_less / model
        folder_marker_based = data_folder_raw_marker_based
        folder_marker_less_fq_corrected = pp_marker_less_folder / "ML_fq_corrected" / model
        folder_marker_less_to_export = pp_marker_less_folder / model

        correct_markerless_fq_folder(folder_marker_based, folder_markerless_to_correct, folder_marker_less_fq_corrected)

        pre_processed_marker_less_folder(
            folder_marker_based=folder_marker_based,
            folder_markerless_to_correct=folder_marker_less_fq_corrected,
            folder_marker_less_to_export=folder_marker_less_to_export,
            remove_nan=True,
            filter_data=True,
            cutoff=6.0,
            order=4,
        )


if __name__ == "__main__":
    # calculate time of processing
    import time

    start_time = time.time()
    # Pre process the marker based data
    pre_process_marker_based_data()
    # Pre process the marker less data
    pre_process_marker_less_data()
    # Fusion of the RTMPOSE and SynthPose data into one file
    fusion_markerless_model(
        path_synthpose=pp_marker_less_folder / "SynthPose",
        path_rtmpose=pp_marker_less_folder / "RTMPose",
        path_export=pp_marker_less_folder / "SynthRTMPose",
    )
    # Fusion of the RTMPOSE and SynthPose data with the marker-based data into one file to check if the marker-based data are coherent with the marker-less data
    fusion_model(
        path_synthpose=pp_marker_less_folder / "SynthRTMPose",
        path_rtmpose=pp_marker_based_folder,
        path_export=pp_marker_less_folder / "SynthRTMPoseMB_Marker",
    )
    stop_time = time.time()
    print(f"Total processing time: {stop_time - start_time:.2f} seconds")
