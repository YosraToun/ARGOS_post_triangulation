from pathlib import Path
import numpy as np
from biobuddy import (
    Axis,
    BiomechanicalModel,
    C3dData,
    Marker,
    Mesh,
    Segment,
    SegmentCoordinateSystem,
    Translations,
    RangeOfMotion,
    Ranges,
    Rotations,
    ViewAs,
    SegmentCoordinateSystemUtils,
)


def fake_rtmpose(
    static_trial_path: Path,
    model_name: str,
    side_to_process: str,
    two_dof_hand_model: bool = False,
    humerus_as_one_segment: bool = False,
    animate_model: bool = True,
):
    
    static_trial = C3dData(str(static_trial_path))
    output_model_filepath = f"{model_name}.bioMod"
    # Generate the personalized kinematic model
    reduced_model = BiomechanicalModel()
    reduced_model.add_segment(Segment(name="Ground"))

    ## define a Ground segment based on the orientation of the thorax () + getting the global positoin of head in this frame
    reduced_model.add_segment(
        Segment(
            name="Thorax",
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            dof_names=["Thorax_TX", "Thorax_TY", "Thorax_TZ", "Thorax_RX", "Thorax_RY", "Thorax_RZ"],
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "IJ"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start="T10",
                    end="C7",
                ),
                second_axis=Axis(name=Axis.Name.X, start="T10", end="IJ"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("IJ", "C7", "T10"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("IJ", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))

    # GH = lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["R_AC"])(static_trial.values, None) - np.array([0.0, 0.00, 0.08, 0.0])
    
    if "right" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["R_Shoulder_AddAbd","R_Shoulder_FleExt","R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="R_EJC", end="R_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_GH", "R_EJC"), is_local=False),
                )
            )
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_GH", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_EJC", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["R_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="R_EJC", end="R_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_GH", "R_EJC"), is_local=False),
                )
            )

            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_2",
                    parent_name="R_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["R_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="R_EJC", end="R_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_GH", "R_EJC"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_3",
                    parent_name="R_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="R_EJC", end="R_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_GH", "R_EJC"), is_local=False),
                )
            )
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_GH", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_EJC", is_technical=True, is_anatomical=True))

        if humerus_as_one_segment:
            parent_name_Rhum = "R_Humerus"
        else:
            parent_name_Rhum = "R_Humerus_3"
        reduced_model.add_segment(
            Segment(
                name="R_Forearm_1",
                parent_name=parent_name_Rhum,
                rotations=Rotations.Z,
                dof_names=["R_Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_EJC",
                    first_axis=Axis(name=Axis.Name.Y, start="R_WJC", end="R_EJC"),
                    second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_GH"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_WJC", "R_EJC"), is_local=False),
            )
        )
        # reduced_model.segments["Forearm"].add_marker(Marker("R_WJC", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["R_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_EJC",
                    first_axis=Axis(name=Axis.Name.Y, start="R_WJC", end="R_EJC"),
                    second_axis=Axis(name=Axis.Name.X, start="R_EJC", end="R_GH"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_WJC", "R_EJC"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_WJC", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["R_WJC_FleExt", "R_WJC_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["R_WJC_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="R_Hand",
                parent_name="R_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_WJC",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_HM2", "R_HM5"]),
                        end="R_WJC",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_HM5", end="R_HM2"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_WJC", "R_HM2", "R_HM5"), is_local=False),
            )
        )
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM2", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM5", is_technical=True, is_anatomical=True))
    if "left" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["L_Shoulder_AddAbd","L_Shoulder_FleExt","L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="L_EJC", end="L_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_GH", "L_EJC"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_GH", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_EJC", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["L_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="L_EJC", end="L_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_GH", "L_EJC"), is_local=False),
                )
            )

            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_2",
                    parent_name="L_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["L_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="L_EJC", end="L_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_GH", "L_EJC"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_3",
                    parent_name="L_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_GH",
                        first_axis=Axis(name=Axis.Name.Y, start="L_EJC", end="L_GH"),
                        second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_WJC"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_GH", "L_EJC"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_GH", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_EJC", is_technical=True, is_anatomical=True))

        if humerus_as_one_segment:
            parent_name_Lhum = "L_Humerus"
        else:
            parent_name_Lhum = "L_Humerus_3"
        reduced_model.add_segment(
            Segment(
                name="L_Forearm_1",
                parent_name=parent_name_Lhum,
                rotations=Rotations.Z,
                dof_names=["L_Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_EJC",
                    first_axis=Axis(name=Axis.Name.Y, start="L_WJC", end="L_EJC"),
                    second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_GH"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_GH", "L_EJC"), is_local=False),
            )
        )
        # reduced_model.segments["Forearm"].add_marker(Marker("R_WJC", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_EJC",
                    first_axis=Axis(name=Axis.Name.Y, start="L_WJC", end="L_EJC"),
                    second_axis=Axis(name=Axis.Name.X, start="L_EJC", end="L_GH"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_WJC", "L_EJC"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_WJC", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["L_WJC_FleExt", "L_WJC_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["L_WJC_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="L_Hand",
                parent_name="L_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_WJC",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_HM2", "L_HM5"]),
                        end="L_WJC",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_HM2", end="L_HM5"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_WJC", "L_HM2", "L_HM5"), is_local=False),
            )
        )
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM2", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM5", is_technical=True, is_anatomical=True))
    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real