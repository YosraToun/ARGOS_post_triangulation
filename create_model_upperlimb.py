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


def marker_based(
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
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "T10", "IJ", "PX"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start=SegmentCoordinateSystemUtils.mean_markers(["PX", "T10"]),
                    end=SegmentCoordinateSystemUtils.mean_markers(["C7", "IJ"]),
                ),
                second_axis=Axis(name=Axis.Name.X, start="C7", end="IJ"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("IJ", "PX", "C7", "T10"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("IJ", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("PX", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))

    if "right" in side_to_process.lower():
        if humerus_as_one_segment:
            reduced_model.add_segment(
                            Segment(
                                name="R_Humerus",
                                parent_name="Thorax",
                                rotations=Rotations.XZY,
                                dof_names=["R_Shoulder_AddAbd","R_Shoulder_FleExt","R_Shoulder_Rot"],
                                segment_coordinate_system=SegmentCoordinateSystem(
                                    origin="R_GH",
                                    first_axis=Axis(
                                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                                    ),
                                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                                    axis_to_keep=Axis.Name.Y,
                                ),
                                mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
                            )
                        )
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_EM", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_EL", is_technical=True, is_anatomical=True))
        else: 
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["R_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_GH",
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
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
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
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
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]), end="R_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_AC", "R_EM", "R_EL"), is_local=False),
                )
            )

            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_EM", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_EL", is_technical=True, is_anatomical=True))
            # reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_Up_tech_arm", is_technical=True, is_anatomical=True))
            # reduced_model.segments["R_Humerus_3"].add_marker(
            #     Marker("R_Med_tech_arm", is_technical=True, is_anatomical=True)
            # )
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
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_EM", end="R_EL"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("R_EL", "R_EL"), is_local=False),
            )
        )
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("R_EL", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("R_EM", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["R_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_EL", "R_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_US", end="R_RS"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_US", "R_RS"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_US", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_RS", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["R_Wrist_FleExt", "R_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["R_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="R_Hand",
                parent_name="R_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_HM2", "R_HM5"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["R_RS", "R_US"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_HM5", end="R_HM2"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_US", "R_RS"), is_local=False),
            )
        )
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM2", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_HM5", is_technical=True, is_anatomical=True))

    if "left" in side_to_process.lower():
        if humerus_as_one_segment:
            reduced_model.add_segment(
                            Segment(
                                name="L_Humerus",
                                parent_name="Thorax",
                                rotations=Rotations.XZY,
                                dof_names=["L_Shoulder_AddAbd","L_Shoulder_FleExt","L_Shoulder_Rot"],
                                segment_coordinate_system=SegmentCoordinateSystem(
                                    origin="L_GH",
                                    first_axis=Axis(
                                        name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                                    ),
                                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                                    axis_to_keep=Axis.Name.Y,
                                ),
                                mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
                            )
                        )
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_EM", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_EL", is_technical=True, is_anatomical=True))
            # reduced_model.segments["L_Humerus"].add_marker(Marker("L_tech_up_arm", is_technical=True, is_anatomical=True))
            # reduced_model.segments["L_Humerus"].add_marker(
            #     Marker("L_tech_med_arm", is_technical=True, is_anatomical=True)
            # )

        else: 
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["L_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_GH",
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
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
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
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
                        first_axis=Axis(
                            name=Axis.Name.Y, start=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]), end="L_GH"
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_AC", "L_EM", "L_EL"), is_local=False),
                )
            )

            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_EM", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_EL", is_technical=True, is_anatomical=True))
            # reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_tech_up_arm", is_technical=True, is_anatomical=True))
            # reduced_model.segments["L_Humerus_3"].add_marker(
            #     Marker("L_tech_med_arm", is_technical=True, is_anatomical=True)
            # )
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
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_EL", end="L_EM"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("L_EL", "L_EM"), is_local=False),
            )
        )
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("L_EL", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("L_EM", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_EL", "L_EM"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_RS", end="L_US"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_US", "L_RS"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_US", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_RS", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["L_Wrist_FleExt", "L_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["L_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="L_Hand",
                parent_name="L_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_HM2", "L_HM5"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["L_RS", "L_US"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_HM2", end="L_HM5"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_US", "L_RS"), is_local=False),
            )
        )
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM2", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_HM5", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real


def synthRTM(
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
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "sternum"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start="T11",
                    end="C7",
                ),
                second_axis=Axis(name=Axis.Name.X, start="C7", end="sternum"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("sternum", "C7", "T11"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("sternum", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T11", is_technical=True, is_anatomical=True))

    if "right" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
            Segment(
                name="R_Humerus",
                parent_name="Thorax",
                rotations=Rotations.XZY,
                dof_names=["R_Shoulder_AddAbd","R_Shoulder_FleExt","R_Shoulder_Rot"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_Shoulder",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                        end="R_Shoulder",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
            ))
            reduced_model.segments["R_Humerus"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))
        else:  
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["R_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_2",
                    parent_name="R_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["R_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_3",
                    parent_name="R_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )

            reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))

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
                    origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start="R_base_hand",
                        end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("r_lelbow", "r_melbow"), is_local=False),
            )
        )
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["R_Forearm_ProSup"],
                # q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start="R_base_hand",
                        end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_MCP_little", end="R_MCP_index"),
                    axis_to_keep=Axis.Name.Y,
                ),
                # mesh=Mesh(("r_lwrist", "r_mwrist"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_base_hand", is_technical=True, is_anatomical=False))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["R_Wrist_FleExt", "R_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["R_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="R_Hand",
                parent_name="R_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_base_hand",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_MCP_index", "R_MCP_little"]),
                        end="R_base_hand",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_MCP_little", end="R_MCP_index"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_base_hand", "R_MCP_little", "R_MCP_index"), is_local=False),
            )
        )
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_base_hand", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_index", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_little", is_technical=True, is_anatomical=True))

    if "left" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["L_Shoulder_AddAbd","L_Shoulder_FleExt","L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["L_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_2",
                    parent_name="L_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["L_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_3",
                    parent_name="L_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )

            reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))

        if humerus_as_one_segment:
            parent_name_Lhum = "L_Humerus"
        else:
            parent_name_Lhum = "L_Humerus_3"
        
        reduced_model.add_segment(
            Segment(
                name="L_Forearm_1",
                parent_name=parent_name_Lhum,
                rotations=Rotations.Z,
                dof_names=["Elbow_FleExt"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start="L_base_hand",
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("l_lelbow", "l_melbow"), is_local=False),
            )
        )
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["L_Forearm_ProSup"],
                #q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start="L_base_hand",
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_MCP_index", end="L_MCP_little"),
                    axis_to_keep=Axis.Name.Y,
                ),
                # mesh=Mesh(("r_lwrist", "r_mwrist"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_base_hand", is_technical=True, is_anatomical=False))
        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["L_Wrist_FleExt", "L_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["L_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="L_Hand",
                parent_name="L_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_base_hand",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_MCP_index", "L_MCP_little"]),
                        end="L_base_hand",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_MCP_index", end="L_MCP_little"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_base_hand", "L_MCP_little", "L_MCP_index"), is_local=False),
            )
        )
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_base_hand", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_index", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_little", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real


def synthpose(
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
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "sternum"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start="T11",
                    end="C7",
                ),
                second_axis=Axis(name=Axis.Name.X, start="T11", end="sternum"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("sternum", "C7", "T11"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("sternum", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T11", is_technical=True, is_anatomical=True))

    # GH = lambda m, bio: SegmentCoordinateSystemUtils.mean_markers(["R_AC"])(static_trial.values, None) - np.array(
    #     [0.0, 0.00, 0.01, 0.0]
    # )
    if "right" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["R_Shoulder_AddAbd","R_Shoulder_FleExt","R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )
            reduced_model.segments["R_Humerus"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))

        else:
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["R_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_2",
                    parent_name="R_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["R_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_3",
                    parent_name="R_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                            end="R_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "r_melbow", "r_lelbow"), is_local=False),
                )
            )

            reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))
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
                    origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["r_lwrist", "r_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="r_melbow", end="r_lelbow"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("r_lelbow", "r_melbow"), is_local=False),
            )
        )
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_lelbow", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Forearm_1"].add_marker(Marker("r_melbow", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                #q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
                dof_names=["R_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["r_lwrist", "r_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["r_lelbow", "r_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="r_mwrist", end="r_lwrist"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("r_lwrist", "r_mwrist"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("r_lwrist", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("r_mwrist", is_technical=True, is_anatomical=True))

    if "left" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["L_Shoulder_AddAbd","L_Shoulder_FleExt","L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["L_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_2",
                    parent_name="L_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["L_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_3",
                    parent_name="L_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(
                            name=Axis.Name.Y,
                            start=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                            end="L_Shoulder",
                        ),
                        second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "l_melbow", "l_lelbow"), is_local=False),
                )
            )

            reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
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
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["l_lwrist", "l_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="l_lelbow", end="l_melbow"),
                    axis_to_keep=Axis.Name.Z,
                ),
                mesh=Mesh(("l_lelbow", "l_melbow"), is_local=False),
            )
        )
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_lelbow", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Forearm_1"].add_marker(Marker("l_melbow", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                #q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[0], max_bound=[np.pi]),
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["l_lwrist", "l_mwrist"]),
                        end=SegmentCoordinateSystemUtils.mean_markers(["l_lelbow", "l_melbow"]),
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="l_lwrist", end="l_mwrist"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("l_lwrist", "l_mwrist"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("l_lwrist", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("l_mwrist", is_technical=True, is_anatomical=True))

    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real

def rtmpose(
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
                origin=SegmentCoordinateSystemUtils.mean_markers(["C7", "sternum"]),
                first_axis=Axis(
                    name=Axis.Name.Y,
                    start="T11",
                    end="C7",
                ),
                second_axis=Axis(name=Axis.Name.X, start="T11", end="sternum"),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("sternum", "C7", "T11"), is_local=False),
        )
    )

    reduced_model.segments["Thorax"].add_marker(Marker("sternum", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
    reduced_model.segments["Thorax"].add_marker(Marker("T11", is_technical=True, is_anatomical=True))

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
                        origin="R_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="R_Elbow", end="R_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "R_Elbow"), is_local=False),
                )
            )
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_Shoulder", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus"].add_marker(Marker("R_Elbow", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["R_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="R_Elbow", end="R_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "R_Elbow"), is_local=False),
                )
            )

            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_2",
                    parent_name="R_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["R_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="R_Elbow", end="R_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "R_Elbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="R_Humerus_3",
                    parent_name="R_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["R_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="R_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="R_Elbow", end="R_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("R_Shoulder", "R_Elbow"), is_local=False),
                )
            )
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_Shoulder", is_technical=True, is_anatomical=True))
            reduced_model.segments["R_Humerus_3"].add_marker(Marker("R_Elbow", is_technical=True, is_anatomical=True))

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
                    origin="R_Elbow",
                    first_axis=Axis(name=Axis.Name.Y, start="R_Wrist", end="R_Elbow"),
                    second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Shoulder"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_Wrist", "R_Elbow"), is_local=False),
            )
        )
        # reduced_model.segments["Forearm"].add_marker(Marker("R_wrist", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="R_Forearm_2",
                parent_name="R_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["R_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_Elbow",
                    first_axis=Axis(name=Axis.Name.Y, start="R_Wrist", end="R_Elbow"),
                    second_axis=Axis(name=Axis.Name.X, start="R_Elbow", end="R_Shoulder"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_Wrist", "R_Elbow"), is_local=False),
            )
        )
        reduced_model.segments["R_Forearm_2"].add_marker(Marker("R_Wrist", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["R_Wrist_FleExt", "R_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["R_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="R_Hand",
                parent_name="R_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="R_Wrist",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["R_MCP_index", "R_MCP_little"]),
                        end="R_Wrist",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="R_MCP_little", end="R_MCP_index"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("R_Wrist", "R_MCP_index", "R_MCP_little"), is_local=False),
            )
        )
        reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_index", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["R_Hand"].add_marker(Marker("R_MCP_little", is_technical=True, is_anatomical=True))
    if "left" in side_to_process:
        if humerus_as_one_segment:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus",
                    parent_name="Thorax",
                    rotations=Rotations.XZY,
                    dof_names=["L_Shoulder_AddAbd","L_Shoulder_FleExt","L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="L_Elbow", end="L_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "L_Elbow"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_Shoulder", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus"].add_marker(Marker("L_Elbow", is_technical=True, is_anatomical=True))
        else:
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_1",
                    parent_name="Thorax",
                    rotations=Rotations.X,
                    dof_names=["L_Shoulder_AddAbd"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="L_Elbow", end="L_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "L_Elbow"), is_local=False),
                )
            )

            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_2",
                    parent_name="L_Humerus_1",
                    rotations=Rotations.Z,
                    dof_names=["L_Shoulder_FleExt"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="L_Elbow", end="L_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "L_Elbow"), is_local=False),
                )
            )
            reduced_model.add_segment(
                Segment(
                    name="L_Humerus_3",
                    parent_name="L_Humerus_2",
                    rotations=Rotations.Y,
                    dof_names=["L_Shoulder_Rot"],
                    segment_coordinate_system=SegmentCoordinateSystem(
                        origin="L_Shoulder",
                        first_axis=Axis(name=Axis.Name.Y, start="L_Elbow", end="L_Shoulder"),
                        second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Wrist"),
                        axis_to_keep=Axis.Name.Y,
                    ),
                    mesh=Mesh(("L_Shoulder", "L_Elbow"), is_local=False),
                )
            )
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_Shoulder", is_technical=True, is_anatomical=True))
            reduced_model.segments["L_Humerus_3"].add_marker(Marker("L_Elbow", is_technical=True, is_anatomical=True))

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
                    origin="L_Elbow",
                    first_axis=Axis(name=Axis.Name.Y, start="L_Wrist", end="L_Elbow"),
                    second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Shoulder"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_Shoulder", "L_Elbow"), is_local=False),
            )
        )
        # reduced_model.segments["Forearm"].add_marker(Marker("R_wrist", is_technical=True, is_anatomical=True))

        reduced_model.add_segment(
            Segment(
                name="L_Forearm_2",
                parent_name="L_Forearm_1",
                rotations=Rotations.Y,
                dof_names=["L_Forearm_ProSup"],
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_Elbow",
                    first_axis=Axis(name=Axis.Name.Y, start="L_Wrist", end="L_Elbow"),
                    second_axis=Axis(name=Axis.Name.X, start="L_Elbow", end="L_Shoulder"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_Wrist", "L_Elbow"), is_local=False),
            )
        )
        reduced_model.segments["L_Forearm_2"].add_marker(Marker("L_Wrist", is_technical=True, is_anatomical=True))

        if two_dof_hand_model:
            rotation_hand = Rotations.ZX
            dof_names_hand = ["L_Wrist_FleExt", "L_Wrist_Dev"]
        else:
            rotation_hand = Rotations.Z
            dof_names_hand = ["L_Wrist_FleExt"]

        reduced_model.add_segment(
            Segment(
                name="L_Hand",
                parent_name="L_Forearm_2",
                rotations=rotation_hand,
                dof_names=dof_names_hand,
                segment_coordinate_system=SegmentCoordinateSystem(
                    origin="L_Wrist",
                    first_axis=Axis(
                        name=Axis.Name.Y,
                        start=SegmentCoordinateSystemUtils.mean_markers(["L_MCP_index", "L_MCP_little"]),
                        end="L_Wrist",
                    ),
                    second_axis=Axis(name=Axis.Name.Z, start="L_MCP_index", end="L_MCP_little"),
                    axis_to_keep=Axis.Name.Y,
                ),
                mesh=Mesh(("L_Wrist", "L_MCP_index", "L_MCP_little"), is_local=False),
            )
        )
        reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_index", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_middle", is_technical=True, is_anatomical=True))
        # reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_ring", is_technical=True, is_anatomical=True))
        reduced_model.segments["L_Hand"].add_marker(Marker("L_MCP_little", is_technical=True, is_anatomical=True))
    # Put the model together, print it and print it to a bioMod file
    model_real = reduced_model.to_real(static_trial)
    model_real.to_biomod(output_model_filepath)

    if animate_model:
        model_real.animate(view_as=ViewAs.BIORBD, model_path=output_model_filepath)

    return model_real