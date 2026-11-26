from pathlib import Path
import ezc3d
import numpy as np
import polars as pl


INPUT_ROOT = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\data\pre_processed\marker_less\SynthRTMPoseMB_Marker"
)
OUTPUT_ROOT = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\reference_frames"
)
MODELS = ("marker_based", "synthrtm", "synthpose", "rtmpose", "fake_rtmpose")


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def build_scs_matrix(
    origin: np.ndarray,
    first_axis_name: str,
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_axis_name: str,
    second_start: np.ndarray,
    second_end: np.ndarray,
    axis_to_keep: str,
) -> np.ndarray:
    """
    Build 4x4 homogeneous local-to-global transform(s) across all N frames.
    Returns shape (N, 4, 4) for (N, 3) inputs or (4, 4) for (3,) inputs.
    """
    origin = np.asarray(origin, dtype=float)
    u1 = _normalize(np.asarray(first_end, dtype=float) - np.asarray(first_start, dtype=float))
    u2 = _normalize(np.asarray(second_end, dtype=float) - np.asarray(second_start, dtype=float))

    keep = axis_to_keep.upper()
    sec = second_axis_name.upper()

    if keep == "Y":
        ey = u1
        if sec == "X":
            ez = _normalize(np.cross(u2, ey))
            ex = _normalize(np.cross(ey, ez))
        elif sec == "Z":
            ex = _normalize(np.cross(ey, u2))
            ez = _normalize(np.cross(ex, ey))
        else:
            raise ValueError(f"Unsupported (keep=Y, second={sec}) combination.")
    elif keep == "Z":
        ez = u1
        if sec == "Z":
            ex = _normalize(np.cross(u2, ez))
            ey = _normalize(np.cross(ez, ex))
        elif sec == "X":
            ey = _normalize(np.cross(ez, u2))
            ex = _normalize(np.cross(ey, ez))
        else:
            raise ValueError(f"Unsupported (keep=Z, second={sec}) combination.")
    elif keep == "X":
        ex = u1
        if sec == "Y":
            ez = _normalize(np.cross(ex, u2))
            ey = _normalize(np.cross(ez, ex))
        elif sec == "Z":
            ey = _normalize(np.cross(u2, ex))
            ez = _normalize(np.cross(ex, ey))
        else:
            raise ValueError(f"Unsupported (keep=X, second={sec}) combination.")
    else:
        raise ValueError(f"Unsupported axis_to_keep: {axis_to_keep}")

    R = np.stack([ex, ey, ez], axis=-1)
    batched = R.ndim == 3
    n = R.shape[0] if batched else 1
    T = np.zeros((n, 4, 4), dtype=float) if batched else np.zeros((4, 4), dtype=float)
    T[..., :3, :3] = R
    T[..., :3, 3] = origin
    T[..., 3, 3] = 1.0
    return T


def load_c3d_markers_meters(c3d_path: Path) -> dict[str, np.ndarray]:
    """Load marker trajectories as {label: (N, 3) in meters}, plus lowercase fallback."""
    c = ezc3d.c3d(str(c3d_path))
    labels = [lbl.strip() for lbl in c["parameters"]["POINT"]["LABELS"]["value"]]
    units = c["parameters"]["POINT"]["UNITS"]["value"]
    scale = 1e-3 if (units and str(units[0]).strip().lower() == "mm") else 1.0

    pts = c["data"]["points"][:3, :, :] * scale
    markers: dict[str, np.ndarray] = {}
    for idx, lbl in enumerate(labels):
        arr = np.moveaxis(pts[:, idx, :], -1, 0)
        markers[lbl] = arr
        markers.setdefault(lbl.lower(), arr)
    return markers


def get_pt(markers: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in markers:
        return markers[name]
    if name.lower() in markers:
        return markers[name.lower()]
    raise KeyError(f"Marker '{name}' not found in C3D file.")


def mean_pts(markers: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.mean(np.stack([get_pt(markers, n) for n in names], axis=0), axis=0)


def build_frames_for_markers(
    markers: dict[str, np.ndarray],
    model_type: str,
    side: str,
) -> dict[str, np.ndarray]:
    """Build {segment_name: (N, 4, 4)} for all frames of a trial."""
    mtype = model_type.lower()
    side_l = side.lower()
    frames: dict[str, np.ndarray] = {}

    if mtype == "marker_based":
        frames["Thorax"] = build_scs_matrix(
            origin=mean_pts(markers, ["C7", "T10", "IJ", "PX"]),
            first_axis_name="Y",
            first_start=mean_pts(markers, ["PX", "T10"]),
            first_end=mean_pts(markers, ["C7", "IJ"]),
            second_axis_name="X",
            second_start=get_pt(markers, "C7"),
            second_end=get_pt(markers, "IJ"),
            axis_to_keep="Y",
        )
        if "right" in side_l:
            frames["R_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "R_GH"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_EL", "R_EM"]),
                first_end=get_pt(markers, "R_GH"),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_EM"),
                second_end=get_pt(markers, "R_EL"),
                axis_to_keep="Y",
            )
            frames["R_Forearm"] = build_scs_matrix(
                origin=mean_pts(markers, ["R_EL", "R_EM"]),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_RS", "R_US"]),
                first_end=mean_pts(markers, ["R_EL", "R_EM"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_EM"),
                second_end=get_pt(markers, "R_EL"),
                axis_to_keep="Z",
            )
            frames["R_Hand"] = build_scs_matrix(
                origin=mean_pts(markers, ["R_RS", "R_US"]),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_HM2", "R_HM5"]),
                first_end=mean_pts(markers, ["R_RS", "R_US"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_HM5"),
                second_end=get_pt(markers, "R_HM2"),
                axis_to_keep="Y",
            )
        if "left" in side_l:
            frames["L_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "L_GH"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_EL", "L_EM"]),
                first_end=get_pt(markers, "L_GH"),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_EL"),
                second_end=get_pt(markers, "L_EM"),
                axis_to_keep="Y",
            )
            frames["L_Forearm"] = build_scs_matrix(
                origin=mean_pts(markers, ["L_EL", "L_EM"]),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_RS", "L_US"]),
                first_end=mean_pts(markers, ["L_EL", "L_EM"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_EL"),
                second_end=get_pt(markers, "L_EM"),
                axis_to_keep="Z",
            )
            frames["L_Hand"] = build_scs_matrix(
                origin=mean_pts(markers, ["L_RS", "L_US"]),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_HM2", "L_HM5"]),
                first_end=mean_pts(markers, ["L_RS", "L_US"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_HM2"),
                second_end=get_pt(markers, "L_HM5"),
                axis_to_keep="Y",
            )

    elif mtype == "synthrtm":
        frames["Thorax"] = build_scs_matrix(
            origin=mean_pts(markers, ["C7", "sternum"]),
            first_axis_name="Y",
            first_start=get_pt(markers, "T11"),
            first_end=get_pt(markers, "C7"),
            second_axis_name="X",
            second_start=get_pt(markers, "C7"),
            second_end=get_pt(markers, "sternum"),
            axis_to_keep="Y",
        )
        if "right" in side_l:
            frames["R_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "R_Shoulder"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["r_lelbow", "r_melbow"]),
                first_end=get_pt(markers, "R_Shoulder"),
                second_axis_name="Z",
                second_start=get_pt(markers, "r_melbow"),
                second_end=get_pt(markers, "r_lelbow"),
                axis_to_keep="Y",
            )
            frames["R_Forearm"] = build_scs_matrix(
                origin=mean_pts(markers, ["r_lelbow", "r_melbow"]),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_base_hand"),
                first_end=mean_pts(markers, ["r_lelbow", "r_melbow"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "r_melbow"),
                second_end=get_pt(markers, "r_lelbow"),
                axis_to_keep="Z",
            )
            frames["R_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "R_base_hand"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_MCP_index", "R_MCP_little"]),
                first_end=get_pt(markers, "R_base_hand"),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_MCP_little"),
                second_end=get_pt(markers, "R_MCP_index"),
                axis_to_keep="Y",
            )
        if "left" in side_l:
            frames["L_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "L_Shoulder"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["l_lelbow", "l_melbow"]),
                first_end=get_pt(markers, "L_Shoulder"),
                second_axis_name="Z",
                second_start=get_pt(markers, "l_lelbow"),
                second_end=get_pt(markers, "l_melbow"),
                axis_to_keep="Y",
            )
            frames["L_Forearm"] = build_scs_matrix(
                origin=mean_pts(markers, ["l_lelbow", "l_melbow"]),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_base_hand"),
                first_end=mean_pts(markers, ["l_lelbow", "l_melbow"]),
                second_axis_name="Z",
                second_start=get_pt(markers, "l_lelbow"),
                second_end=get_pt(markers, "l_melbow"),
                axis_to_keep="Z",
            )
            frames["L_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "L_base_hand"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_MCP_index", "L_MCP_little"]),
                first_end=get_pt(markers, "L_base_hand"),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_MCP_index"),
                second_end=get_pt(markers, "L_MCP_little"),
                axis_to_keep="Y",
            )

    elif mtype == "synthpose":
        frames["Thorax"] = build_scs_matrix(
            origin=mean_pts(markers, ["C7", "sternum"]),
            first_axis_name="Y",
            first_start=get_pt(markers, "T11"),
            first_end=get_pt(markers, "C7"),
            second_axis_name="X",
            second_start=get_pt(markers, "T11"),
            second_end=get_pt(markers, "sternum"),
            axis_to_keep="Y",
        )
        if "right" in side_l:
            frames["R_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "R_Shoulder"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_elbow"),
                first_end=get_pt(markers, "R_Shoulder"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_elbow"),
                second_end=get_pt(markers, "R_wrist"),
                axis_to_keep="Y",
            )
            frames["R_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "R_elbow"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_wrist"),
                first_end=get_pt(markers, "R_elbow"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_elbow"),
                second_end=get_pt(markers, "R_Shoulder"),
                axis_to_keep="Y",
            )
        if "left" in side_l:
            frames["L_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "L_Shoulder"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_elbow"),
                first_end=get_pt(markers, "L_Shoulder"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_elbow"),
                second_end=get_pt(markers, "L_wrist"),
                axis_to_keep="Y",
            )
            frames["L_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "L_elbow"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_wrist"),
                first_end=get_pt(markers, "L_elbow"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_elbow"),
                second_end=get_pt(markers, "L_Shoulder"),
                axis_to_keep="Y",
            )

    elif mtype == "rtmpose":
        frames["Thorax"] = build_scs_matrix(
            origin=mean_pts(markers, ["C7", "sternum"]),
            first_axis_name="Y",
            first_start=get_pt(markers, "T11"),
            first_end=get_pt(markers, "C7"),
            second_axis_name="X",
            second_start=get_pt(markers, "T11"),
            second_end=get_pt(markers, "sternum"),
            axis_to_keep="Y",
        )
        if "right" in side_l:
            frames["R_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "R_Shoulder"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_elbow"),
                first_end=get_pt(markers, "R_Shoulder"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_elbow"),
                second_end=get_pt(markers, "R_wrist"),
                axis_to_keep="Y",
            )
            frames["R_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "R_elbow"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_wrist"),
                first_end=get_pt(markers, "R_elbow"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_elbow"),
                second_end=get_pt(markers, "R_Shoulder"),
                axis_to_keep="Y",
            )
            frames["R_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "R_wrist"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_MCP_index", "R_MCP_little"]),
                first_end=get_pt(markers, "R_wrist"),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_MCP_little"),
                second_end=get_pt(markers, "R_MCP_index"),
                axis_to_keep="Y",
            )
        if "left" in side_l:
            frames["L_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "L_Shoulder"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_elbow"),
                first_end=get_pt(markers, "L_Shoulder"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_elbow"),
                second_end=get_pt(markers, "L_wrist"),
                axis_to_keep="Y",
            )
            frames["L_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "L_elbow"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_wrist"),
                first_end=get_pt(markers, "L_elbow"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_elbow"),
                second_end=get_pt(markers, "L_Shoulder"),
                axis_to_keep="Y",
            )
            frames["L_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "L_wrist"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_MCP_index", "L_MCP_little"]),
                first_end=get_pt(markers, "L_wrist"),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_MCP_index"),
                second_end=get_pt(markers, "L_MCP_little"),
                axis_to_keep="Y",
            )

    elif mtype == "fake_rtmpose":
        frames["Thorax"] = build_scs_matrix(
            origin=mean_pts(markers, ["C7", "IJ"]),
            first_axis_name="Y",
            first_start=get_pt(markers, "T10"),
            first_end=get_pt(markers, "C7"),
            second_axis_name="X",
            second_start=get_pt(markers, "T10"),
            second_end=get_pt(markers, "IJ"),
            axis_to_keep="Y",
        )
        if "right" in side_l:
            frames["R_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "R_GH"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_EJC"),
                first_end=get_pt(markers, "R_GH"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_EJC"),
                second_end=get_pt(markers, "R_WJC"),
                axis_to_keep="Y",
            )
            frames["R_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "R_EJC"),
                first_axis_name="Y",
                first_start=get_pt(markers, "R_WJC"),
                first_end=get_pt(markers, "R_EJC"),
                second_axis_name="X",
                second_start=get_pt(markers, "R_EJC"),
                second_end=get_pt(markers, "R_GH"),
                axis_to_keep="Y",
            )
            frames["R_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "R_WJC"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["R_HM2", "R_HM5"]),
                first_end=get_pt(markers, "R_WJC"),
                second_axis_name="Z",
                second_start=get_pt(markers, "R_HM5"),
                second_end=get_pt(markers, "R_HM2"),
                axis_to_keep="Y",
            )
        if "left" in side_l:
            frames["L_Humerus"] = build_scs_matrix(
                origin=get_pt(markers, "L_GH"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_EJC"),
                first_end=get_pt(markers, "L_GH"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_EJC"),
                second_end=get_pt(markers, "L_WJC"),
                axis_to_keep="Y",
            )
            frames["L_Forearm"] = build_scs_matrix(
                origin=get_pt(markers, "L_EJC"),
                first_axis_name="Y",
                first_start=get_pt(markers, "L_WJC"),
                first_end=get_pt(markers, "L_EJC"),
                second_axis_name="X",
                second_start=get_pt(markers, "L_EJC"),
                second_end=get_pt(markers, "L_GH"),
                axis_to_keep="Y",
            )
            frames["L_Hand"] = build_scs_matrix(
                origin=get_pt(markers, "L_WJC"),
                first_axis_name="Y",
                first_start=mean_pts(markers, ["L_HM2", "L_HM5"]),
                first_end=get_pt(markers, "L_WJC"),
                second_axis_name="Z",
                second_start=get_pt(markers, "L_HM2"),
                second_end=get_pt(markers, "L_HM5"),
                axis_to_keep="Y",
            )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return frames


def frames_to_dataframe(
    frames: dict[str, np.ndarray],
    model_type: str,
    subject_id: str,
    trial_name: str,
) -> pl.DataFrame:
    """Convert {segment: (N, 4, 4)} into a tidy Polars DataFrame."""
    rows: list[pl.DataFrame] = []
    for seg_name, T in frames.items():
        n_frames = T.shape[0]
        R = T[:, :3, :3]
        o = T[:, :3, 3]
        ex, ey, ez = R[:, :, 0], R[:, :, 1], R[:, :, 2]

        rows.append(
            pl.DataFrame(
                {
                    "model": [model_type] * n_frames,
                    "subject": [subject_id] * n_frames,
                    "trial": [trial_name] * n_frames,
                    "segment": [seg_name] * n_frames,
                    "frame": np.arange(n_frames, dtype=np.int32),
                    "origin_x": o[:, 0],
                    "origin_y": o[:, 1],
                    "origin_z": o[:, 2],
                    "ex_x": ex[:, 0],
                    "ex_y": ex[:, 1],
                    "ex_z": ex[:, 2],
                    "ey_x": ey[:, 0],
                    "ey_y": ey[:, 1],
                    "ey_z": ey[:, 2],
                    "ez_x": ez[:, 0],
                    "ez_y": ez[:, 1],
                    "ez_z": ez[:, 2],
                }
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame()


def run_batch(
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    models: tuple[str, ...] = MODELS,
    side: str = "both",
    skip_missing_markers: bool = True,
) -> None:
    subject_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        c3d_files = sorted(subj_dir.rglob("*.c3d"))

        for c3d_path in c3d_files:
            markers = load_c3d_markers_meters(c3d_path)
            trial_rel = c3d_path.relative_to(subj_dir).with_suffix(".parquet")

            for model_type in models:
                out_path = output_root / model_type / subject_id / trial_rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    frames = build_frames_for_markers(markers, model_type=model_type, side=side)
                    df = frames_to_dataframe(
                        frames=frames,
                        model_type=model_type,
                        subject_id=subject_id,
                        trial_name=c3d_path.stem,
                    )
                    df.write_parquet(out_path)
                except KeyError as exc:
                    if skip_missing_markers:
                        print(f"[SKIP] {model_type} | {subject_id} | {c3d_path.name}: {exc}")
                    else:
                        raise


if __name__ == "__main__":
    run_batch(side="both")