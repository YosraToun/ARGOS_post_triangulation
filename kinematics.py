from pathlib import Path
import numpy as np
import pandas as pd
from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
import biorbd
from biobuddy import C3dData


def main(filename, model_name, show: bool = True, filename_output=None):
    # Load a predefined model
    model = biorbd.Model(model_name)
    trial = C3dData(filename)
    nb_frames = trial.nb_frames
    name_dofs = [model.nameDof()[i].to_string() for i in range(model.nbQ())]
    dict_dof = {}
    for ind, name in enumerate(name_dofs):
        dict_dof[name] = ind

    markerNames = [model.markerNames()[i].to_string() for i in range(len(model.markerNames()))]
    markers = trial.get_position(markerNames)[:3, :, :]
    check_nan = np.isnan(markers)
    # check wich frame have NaN values in the markers
    frames_with_nan = np.any(check_nan, axis=(0, 1))
    # transform the markers to remove the NaN values by removing the value
    markers_no_nan = markers[:, :, ~frames_with_nan]
    # bounds = np.array([[-np.pi] * model.nbQ(), [np.pi] * model.nbQ()])
    IK = biorbd.InverseKinematics(model, markers_no_nan)
    # IK.bounds = bounds
    q_recons = IK.solve(method="trf")# TODO : Update biorbd and try proxqp new method to go faster. 
    sol = IK.sol()
    ## Export all kinematics in a csv file
    # find all the keys of dict_dof
    # reposition all the point considering that some were removed because of NaN values, so we need to add the NaN values back to the q_recons array
    q_recons_full = np.full((model.nbQ(), nb_frames), np.nan)
    q_visualisation = np.zeros((model.nbQ(), nb_frames), np.float64)
    q_recons_full[:, ~frames_with_nan] = q_recons
    q_visualisation[:, ~frames_with_nan] = q_recons
    q_2_export = q_recons_full[[dict_dof[name] for name in dict_dof.keys()]]
    # Compute time vector based on frame count and sampling frequency
    # frame = np.arange(first_frame, last_frame, dtype=int)
    frame = np.arange(0, nb_frames, dtype=int)

    # export a csv of the kinematic and the first frame time
    # Extract all value with their names values in a dataframe

    # Create DataFrame
    df = pd.DataFrame({"TimeFrame": frame, **{name: q_2_export[dict_dof[name], :] for name in dict_dof.keys()}})

    # Export to CSV in the same directory as the input file
    name_csv = Path(filename).with_suffix(".csv") if filename_output is None else Path(filename_output)
    # check if the folder of the output file exist, if not create it
    if not name_csv.parent.exists():
        name_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(name_csv, index=False)
    print(f"Exported kinematics to {name_csv}")

    # save q_visualisation as a npy file in the same folder as the output csv file
    if filename_output is not None:
        np.save(
            filename_output.replace(".csv", ".npy"),
            {
                "q": q_visualisation,
                "model_path": str(Path(model_name)),
                "filename_path": str(Path(filename)),
                "optim_information": sol,
            },
            allow_pickle=True,
        )
        print(f"Exported kinematics to {filename_output.replace('.csv', '.npy')}")

    if show:
        nb_seconds = 10
        t_span = np.linspace(0, nb_seconds, nb_frames)
        model.name = "toto"
        model.markerNames
        model.nb_markers = len(model.markerNames())
        model.marker_names = model.markerNames()
        # model.options.show_gravity = False
        model_rerun = BiorbdModel(model_name)
        viz = PhaseRerun(t_span)
        viz.add_animated_model(
            model_rerun, q_recons, tracked_markers=PyoMarkers(data=markers, channels=markerNames, show_labels=False)
        )
        viz.rerun("msk_model")
        # pause to see the model
        input("Press Enter to continue...")
