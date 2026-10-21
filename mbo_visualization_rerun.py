from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
import numpy as np
from biobuddy import C3dData
import biorbd
from pathlib import Path
from choose_file import choose_file_npy


def visualize(npy_file):

    data = np.load(npy_file, allow_pickle=True).item()

    q = data["q"]
    model_name = data["model_path"]
    filename = data["filename_path"]
    # to avoid error in Pyorerun this line was necessary
    model_name = model_name.replace("\\", "/")

    model = biorbd.Model(model_name)
    trial = C3dData(filename)
    print(model_name)
    print(filename)
    print(model_name)
    print(type(model_name))

    markerNames = [model.markerNames()[i].to_string() for i in range(len(model.markerNames()))]
    markers = trial.get_position(markerNames)[:3, :, :]

    nb_frames = q.shape[1]
    t_span = np.linspace(0, 10, nb_frames)

    model_rerun = BiorbdModel(model_name)

    print("Model path:", model_name)
    print("Model stem:", Path(model_name).stem)

    viz = PhaseRerun(t_span)
    viz.add_animated_model(
        model_rerun,
        q,
        tracked_markers=PyoMarkers(
            data=markers,
            channels=markerNames,
            show_labels=True,
        ),
    )

    viz.rerun()
    input("Press Enter to close the viewer and exit...")

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        npy_file = sys.argv[1]
    else:
        npy_file = choose_file_npy()

    if npy_file:
        visualize(npy_file)
        input("Visualization running. Press [Enter] in the terminal to close the window!")