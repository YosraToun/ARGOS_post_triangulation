from pathlib import Path
import create_model_upperlimb
import kinematics




def mbo(
    root_folder_marker_less: Path,
    static_filename: str,
    model_creation,
    folder_to_save: Path,
    two_dof_hand_model: bool = True,
    humerus_as_one_segment: bool = False,
    show_animation: bool = False,
    list_subjects: list = None,
    list_tasks: list = None,
    side_to_process: str = "right_and_left",
):
    for subject_folder in root_folder_marker_less.iterdir():
        if not subject_folder.is_dir():
            continue

        if list_subjects is not None and subject_folder.name not in list_subjects:
            print(f"Skipping subject {subject_folder.name} as it's not in the list of subjects to process.")
            continue
        print(f"\n--- Processing subject: {subject_folder.name} ---")

        # ------------------------------------------------------------
        # 1. Locate static trial
        # ------------------------------------------------------------
        static_trial = None
        for file in subject_folder.glob("*.c3d"):
            if static_filename in file.name.lower():
                static_trial = file
                break

        if static_trial is None:
            print(f"❌ No static trial found for {subject_folder.name}")
            continue

        print(f"✔ Static trial found: {static_trial.name}")

        
        # ------------------------------------------------------------
        # 2. Create model
        # ------------------------------------------------------------
        if "left" in side_to_process.lower():
            model_name_L = folder_to_save / subject_folder.name / "model_L"
                    # check if the folder of the output file exist, if not create it
            if not model_name_L.parent.exists():
                model_name_L.parent.mkdir(parents=True, exist_ok=True)
            print("⚙ Creating model...")
            model_creation(
                static_trial_path=str(static_trial),
                model_name=str(model_name_L),
                side_to_process="left",
                two_dof_hand_model=two_dof_hand_model,
                humerus_as_one_segment=humerus_as_one_segment,
                animate_model=show_animation,  # You can set True if you want visualization
            )
            print(f"✔ Model created: {model_name_L.name}")
            model_name_L = folder_to_save / subject_folder.name / "model_L.bioMod"

        if "right" in side_to_process.lower():
            model_name_R = folder_to_save / subject_folder.name / "model_R"
                    # Right
            if not model_name_R.parent.exists():
                model_name_R.parent.mkdir(parents=True, exist_ok=True)
            model_creation(
                static_trial_path=str(static_trial),
                model_name=str(model_name_R),
                side_to_process="right",
                two_dof_hand_model=two_dof_hand_model,
                humerus_as_one_segment=humerus_as_one_segment,
                animate_model=show_animation,  # You can set True if you want visualization
            )
            print(f"✔ Model created: {model_name_R.name}")
            model_name_R = folder_to_save / subject_folder.name / "model_R.bioMod"
        
        if "both" in side_to_process.lower():
            model_name_RL = folder_to_save / subject_folder.name / "model_RL"
            # Both
            if not model_name_RL.parent.exists():
                model_name_RL.parent.mkdir(parents=True, exist_ok=True)

            model_creation(
                static_trial_path=str(static_trial),
                model_name=str(model_name_RL),
                side_to_process="right_and_left",
                two_dof_hand_model=two_dof_hand_model,
                humerus_as_one_segment=humerus_as_one_segment,
                animate_model=show_animation,  # You can set True if you want visualization
            )
            print(f"✔ Model created: {model_name_RL.name}")
            model_name_RL = folder_to_save / subject_folder.name / "model_RL.bioMod"
        # ------------------------------------------------------------
        # 3. Process all task trials
        # ------------------------------------------------------------
        for task_file in subject_folder.glob("*.c3d"):
            if static_filename in task_file.name.lower():
                continue  # skip static trial
            print(list_tasks)
            if list_tasks is not None and task_file.name not in list_tasks:
                print(f"Skipping task {task_file.name} as it's not in the list of tasks to process.")
                continue

            
            print(f"⚙ Processing task: {task_file.name}")
            if "left" in side_to_process.lower():
                file_exported_L = folder_to_save / subject_folder.name / f"{task_file.stem}_L.csv"
                # check if the folder of the output file exist, if not create it
                if not file_exported_L.parent.exists():
                    file_exported_L.parent.mkdir(parents=True, exist_ok=True)
                kinematics.main(
                    str(task_file),
                    str(model_name_L),
                    show=show_animation,  # Set True if you want to visualize the kinematics results
                    filename_output=str(file_exported_L),
                )
            if "right" in side_to_process.lower():
                file_exported_R = folder_to_save / subject_folder.name / f"{task_file.stem}_R.csv"
                # check if the folder of the output file exist, if not create it
                if not file_exported_R.parent.exists():
                    file_exported_R.parent.mkdir(parents=True, exist_ok=True)
                kinematics.main(
                    str(task_file),
                    str(model_name_R),
                    show=show_animation,  # Set True if you want to visualize the kinematics results
                    filename_output=str(file_exported_R),
                )
            if "both" in side_to_process.lower():
                file_exported_RL = folder_to_save / subject_folder.name / f"{task_file.stem}_RL.csv"
                # check if the folder of the output file exist, if not create it
                if not file_exported_RL.parent.exists():
                    file_exported_RL.parent.mkdir(parents=True, exist_ok=True)
                kinematics.main(
                    str(task_file),
                    str(model_name_RL),
                    show=show_animation,  # Set True if you want to visualize the kinematics results
                    filename_output=str(file_exported_RL),
                )
                print(f"✔ Finished: {task_file.name}")


if __name__ == "__main__":
    # Example usage
    base_save_folder = Path(r".\Kinematics_SB2026_all")  # Adjust this path as needed
    two_dof_hand_model = True  # Set to True if you want to model the hand with 2 DOF (flexion/extension + deviation), False for 1 DOF (flexion/extension only)
    humerus_as_one_segment = True  # Set to True if you want to model the humerus as one segment, False for three segments
    static_filename = "04-take-in-box.c3d"  # Adjust if your static file has a different name

    list_subjects_to_process = None # ["S08", "S09", "S10"]  # Example: Only process subjects S08, S09, and S10
    list_tasks_to_process = None# Example: Only process these specific tasks
    side_to_process = "both"  # Options: string containing "left", "right", or "both" to process left, right, or both sides respectively
    # Synthpose ==> ici j'utilise aussi les données de Synthpose pour le thorax et rtm pour le reste
    root_folder_RTMPOSE= Path(r".\data\pre_processed\marker_less\SynthRTMPose")
    folder_to_save_RTMPOSE = (
        base_save_folder / "Rtmpose" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "Rtmpose" / "with_1dof_hand"
    )

    model_creation = create_model_upperlimb.rtmpose 
    mbo(
        root_folder_RTMPOSE,
        static_filename,
        model_creation,
        folder_to_save_RTMPOSE,
        two_dof_hand_model=two_dof_hand_model,
        humerus_as_one_segment=humerus_as_one_segment,
        show_animation=False,
        list_subjects=list_subjects_to_process,
        list_tasks=list_tasks_to_process,
        side_to_process=side_to_process,
    ) # Use the appropriate function for model creation



    # Marker based
    root_folder_marker_based = Path(r".\data\pre_processed\marker_based")
    folder_to_save_marker_based = (
        base_save_folder / "Marker_based" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "Marker_based" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.marker_based  # Use the appropriate function for model creation
    mbo(
        root_folder_marker_based,
        static_filename,
        model_creation,
        folder_to_save_marker_based,
        two_dof_hand_model=two_dof_hand_model,
        humerus_as_one_segment=humerus_as_one_segment,
        show_animation=False,
        list_subjects=list_subjects_to_process,
        list_tasks=list_tasks_to_process,
        side_to_process=side_to_process,
    )

    # SynthRTMPose
    root_folder_SynthRTMPose = Path(r".\data\pre_processed\marker_less\SynthRTMPose")
    folder_to_save_SynthRTMPose = (
        base_save_folder / "SynthRTMPose" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "SynthRTMPose" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.synthRTM  # Use the appropriate function for model creation
    mbo(
        root_folder_SynthRTMPose,
        static_filename,
        model_creation,
        folder_to_save_SynthRTMPose,
        two_dof_hand_model=two_dof_hand_model,
        humerus_as_one_segment=humerus_as_one_segment,
        show_animation=False,
        list_subjects=list_subjects_to_process,
        list_tasks=list_tasks_to_process,
        side_to_process=side_to_process,
    )

    # Synthpose
    root_folder_SynthPose = Path(r".\data\pre_processed\marker_less\SynthPose")
    folder_to_save_SynthPose = (
        base_save_folder / "Synthpose" / "with_2dof_hand"
        if two_dof_hand_model
        else base_save_folder / "Synthpose" / "with_1dof_hand"
    )
    model_creation = create_model_upperlimb.synthpose  # Use the appropriate function for model creation
    mbo(
        root_folder_SynthPose,
        static_filename,
        model_creation,
        folder_to_save_SynthPose,
        two_dof_hand_model=two_dof_hand_model,
        humerus_as_one_segment=humerus_as_one_segment,
        show_animation=False,
        list_subjects=list_subjects_to_process,
        list_tasks=list_tasks_to_process,
        side_to_process=side_to_process,
    )

