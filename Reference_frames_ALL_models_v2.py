"""
Compute segment reference frames (origin + rotation matrix, global frame) directly
from c3d marker trajectories, without biorbd / biobuddy.

Same segment definitions as create_model_upperlimb.py (origin, first_axis,
second_axis, axis_to_keep), evaluated frame by frame.

Input  : INPUT_ROOT/<subject>/<trial>.c3d            (static + dynamic trials)
Output : OUTPUT_ROOT/<model>/<subject>/<trial>.parquet   for each model in MODELS

Parquet layout: one row per frame, columns
    frame, time,
    <Segment>_O_x, <Segment>_O_y, <Segment>_O_z,                 origin (global)
    <Segment>_R_xx, _xy, _xz, _yx, ... _zz                       rotation matrix, R[i, j] -> _{i}{j}
The columns of R are the segment X, Y, Z axes expressed in the global frame.
Segments: Thorax, R_Humerus, R_Forearm, R_Hand, L_Humerus, L_Forearm, L_Hand
(Forearm = the "Forearm_2" frame of the biomod, i.e. the one carrying pro/supination;
 synthpose has no hand definition, so no Hand columns for it.)
"""
import argparse
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd

INPUT_ROOT = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\data\pre_processed\marker_less\SynthRTMPoseMB_Marker"
)
OUTPUT_ROOT = INPUT_ROOT.parent / "reference_frames"
MODELS = ("marker_based", "synthrtm", "synthpose", "rtmpose", "fake_rtmpose")
TO_METERS = True  # convert marker positions to meters if the c3d units are mm

# ----------------------------------------------------------------------------
# Segment definitions (generated from create_model_upperlimb.py)
#   origin / start / end : marker name, or list of names -> mean of those markers
#   first / second       : (axis name, start, end)  -> axis vector = end - start
#   keep                 : axis kept exactly; the other one is recomputed orthogonal
# Marker names are matched EXACTLY (case-sensitive). Only if no exact match exists and a single
# label matches ignoring case, that label is used and a warning is printed.
# ----------------------------------------------------------------------------
MODEL_DEFS = {
    "marker_based": {
        "Thorax": dict(
            origin=['C7', 'T10', 'IJ', 'PX'],
            first=('Y', ['PX', 'T10'], ['C7', 'IJ']),
            second=('X', 'C7', 'IJ'),
            keep="Y",
        ),
        "R_Humerus": dict(
            origin='R_GH',
            first=('Y', ['R_EL', 'R_EM'], 'R_GH'),
            second=('Z', 'R_EM', 'R_EL'),
            keep="Y",
        ),
        "R_Forearm": dict(
            origin=['R_EL', 'R_EM'],
            first=('Y', ['R_RS', 'R_US'], ['R_EL', 'R_EM']),
            second=('Z', 'R_US', 'R_RS'),
            keep="Y",
        ),
        "R_Hand": dict(
            origin=['R_RS', 'R_US'],
            first=('Y', ['R_HM2', 'R_HM5'], ['R_RS', 'R_US']),
            second=('Z', 'R_HM5', 'R_HM2'),
            keep="Y",
        ),
        "L_Humerus": dict(
            origin='L_GH',
            first=('Y', ['L_EL', 'L_EM'], 'L_GH'),
            second=('Z', 'L_EL', 'L_EM'),
            keep="Y",
        ),
        "L_Forearm": dict(
            origin=['L_EL', 'L_EM'],
            first=('Y', ['L_RS', 'L_US'], ['L_EL', 'L_EM']),
            second=('Z', 'L_RS', 'L_US'),
            keep="Y",
        ),
        "L_Hand": dict(
            origin=['L_RS', 'L_US'],
            first=('Y', ['L_HM2', 'L_HM5'], ['L_RS', 'L_US']),
            second=('Z', 'L_HM2', 'L_HM5'),
            keep="Y",
        ),
    },
    "synthrtm": {
        "Thorax": dict(
            origin=['C7', 'sternum'],
            first=('Y', 'T11', 'C7'),
            second=('X', 'C7', 'sternum'),
            keep="Y",
        ),
        "R_Humerus": dict(
            origin='R_Shoulder',
            first=('Y', ['r_lelbow', 'r_melbow'], 'R_Shoulder'),
            second=('Z', 'r_melbow', 'r_lelbow'),
            keep="Y",
        ),
        "R_Forearm": dict(
            origin=['r_lelbow', 'r_melbow'],
            first=('Y', 'R_base_hand', ['r_lelbow', 'r_melbow']),
            second=('Z', 'R_MCP_little', 'R_MCP_index'),
            keep="Y",
        ),
        "R_Hand": dict(
            origin='R_base_hand',
            first=('Y', ['R_MCP_index', 'R_MCP_little'], 'R_base_hand'),
            second=('Z', 'R_MCP_little', 'R_MCP_index'),
            keep="Y",
        ),
        "L_Humerus": dict(
            origin='L_Shoulder',
            first=('Y', ['l_lelbow', 'l_melbow'], 'L_Shoulder'),
            second=('Z', 'l_lelbow', 'l_melbow'),
            keep="Y",
        ),
        "L_Forearm": dict(
            origin=['l_lelbow', 'l_melbow'],
            first=('Y', 'L_base_hand', ['l_lelbow', 'l_melbow']),
            second=('Z', 'L_MCP_index', 'L_MCP_little'),
            keep="Y",
        ),
        "L_Hand": dict(
            origin='L_base_hand',
            first=('Y', ['L_MCP_index', 'L_MCP_little'], 'L_base_hand'),
            second=('Z', 'L_MCP_index', 'L_MCP_little'),
            keep="Y",
        ),
    },
    "synthpose": {
        "Thorax": dict(
            origin=['C7', 'sternum'],
            first=('Y', 'T11', 'C7'),
            second=('X', 'T11', 'sternum'),
            keep="Y",
        ),
        "R_Humerus": dict(
            origin='R_Shoulder',
            first=('Y', ['r_lelbow', 'r_melbow'], 'R_Shoulder'),
            second=('Z', 'r_melbow', 'r_lelbow'),
            keep="Y",
        ),
        "R_Forearm": dict(
            origin=['r_lelbow', 'r_melbow'],
            first=('Y', ['r_lwrist', 'r_mwrist'], ['r_lelbow', 'r_melbow']),
            second=('Z', 'r_mwrist', 'r_lwrist'),
            keep="Y",
        ),
        # R_Hand: not defined for this model
        "L_Humerus": dict(
            origin='L_Shoulder',
            first=('Y', ['l_lelbow', 'l_melbow'], 'L_Shoulder'),
            second=('Z', 'l_lelbow', 'l_melbow'),
            keep="Y",
        ),
        "L_Forearm": dict(
            origin=['l_lelbow', 'l_melbow'],
            first=('Y', ['l_lwrist', 'l_mwrist'], ['l_lelbow', 'l_melbow']),
            second=('Z', 'l_lwrist', 'l_mwrist'),
            keep="Y",
        ),
        # L_Hand: not defined for this model
    },
    "rtmpose": {
        "Thorax": dict(
            origin=['C7', 'sternum'],
            first=('Y', 'T11', 'C7'),
            second=('X', 'T11', 'sternum'),
            keep="Y",
        ),
        "R_Humerus": dict(
            origin='R_Shoulder',
            first=('Y', 'R_elbow', 'R_Shoulder'),
            second=('X', 'R_elbow', 'R_wrist'),
            keep="Y",
        ),
        "R_Forearm": dict(
            origin='R_elbow',
            first=('Y', 'R_wrist', 'R_elbow'),
            second=('X', 'R_elbow', 'R_Shoulder'),
            keep="Y",
        ),
        "R_Hand": dict(
            origin='R_wrist',
            first=('Y', ['R_MCP_index', 'R_MCP_little'], 'R_wrist'),
            second=('Z', 'R_MCP_little', 'R_MCP_index'),
            keep="Y",
        ),
        "L_Humerus": dict(
            origin='L_Shoulder',
            first=('Y', 'L_Elbow', 'L_Shoulder'),
            second=('X', 'L_Elbow', 'L_wrist'),
            keep="Y",
        ),
        "L_Forearm": dict(
            origin='L_elbow',
            first=('Y', 'L_wrist', 'L_elbow'),
            second=('X', 'L_elbow', 'L_Shoulder'),
            keep="Y",
        ),
        "L_Hand": dict(
            origin='L_Wrist',
            first=('Y', ['L_MCP_index', 'L_MCP_little'], 'L_Wrist'),
            second=('Z', 'L_MCP_index', 'L_MCP_little'),
            keep="Y",
        ),
    },
    "fake_rtmpose": {
        "Thorax": dict(
            origin=['C7', 'IJ'],
            first=('Y', 'T10', 'C7'),
            second=('X', 'T10', 'IJ'),
            keep="Y",
        ),
        "R_Humerus": dict(
            origin='R_GH',
            first=('Y', 'R_EJC', 'R_GH'),
            second=('X', 'R_EJC', 'R_WJC'),
            keep="Y",
        ),
        "R_Forearm": dict(
            origin='R_EJC',
            first=('Y', 'R_WJC', 'R_EJC'),
            second=('X', 'R_EJC', 'R_GH'),
            keep="Y",
        ),
        "R_Hand": dict(
            origin='R_WJC',
            first=('Y', ['R_HM2', 'R_HM5'], 'R_WJC'),
            second=('Z', 'R_HM5', 'R_HM2'),
            keep="Y",
        ),
        "L_Humerus": dict(
            origin='L_GH',
            first=('Y', 'L_EJC', 'L_GH'),
            second=('X', 'L_EJC', 'L_WJC'),
            keep="Y",
        ),
        "L_Forearm": dict(
            origin='L_EJC',
            first=('Y', 'L_WJC', 'L_EJC'),
            second=('X', 'L_EJC', 'L_GH'),
            keep="Y",
        ),
        "L_Hand": dict(
            origin='L_WJC',
            first=('Y', ['L_HM2', 'L_HM5'], 'L_WJC'),
            second=('Z', 'L_HM2', 'L_HM5'),
            keep="Y",
        ),
    },
}

SEGMENT_ORDER = ["Thorax", "R_Humerus", "R_Forearm", "R_Hand", "L_Humerus", "L_Forearm", "L_Hand"]
AXIS_IDX = {"X": 0, "Y": 1, "Z": 2}
R_NAMES = [f"R_{i}{j}" for i in "xyz" for j in "xyz"]


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def _unit(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _point(spec, markers):
    if isinstance(spec, str):
        return markers[spec]
    return np.mean([markers[s] for s in spec], axis=0)


def _markers_needed(defn):
    names = []
    for spec in (defn["origin"], defn["first"][1], defn["first"][2], defn["second"][1], defn["second"][2]):
        names += [spec] if isinstance(spec, str) else list(spec)
    return set(names)


def compute_frame(defn, markers):
    """Return origin (n, 3) and rotation matrix (n, 3, 3) for one segment definition."""
    origin = _point(defn["origin"], markers)
    n1, s1, e1 = defn["first"]
    n2, s2, e2 = defn["second"]
    v1 = _point(e1, markers) - _point(s1, markers)
    v2 = _point(e2, markers) - _point(s2, markers)

    if defn["keep"] == n1:
        a_name, a, b_name, b = n1, v1, n2, v2
    else:
        a_name, a, b_name, b = n2, v2, n1, v1
    c_name = ({"X", "Y", "Z"} - {a_name, b_name}).pop()

    a = _unit(a)
    if (AXIS_IDX[b_name] - AXIS_IDX[a_name]) % 3 == 1:  # (a, b, c) is a cyclic permutation of (X, Y, Z)
        c = _unit(np.cross(a, b))
        b = np.cross(c, a)
    else:
        c = _unit(np.cross(b, a))
        b = np.cross(a, c)
    b = _unit(b)

    axes = {a_name: a, b_name: b, c_name: c}
    rot = np.stack([axes["X"], axes["Y"], axes["Z"]], axis=-1)  # columns = segment axes in global
    return origin, rot


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_c3d(path, to_meters=True):
    c = ezc3d.c3d(str(path))
    params = c["parameters"]["POINT"]
    labels = [str(l).strip().split(":")[-1] for l in params["LABELS"]["value"]]
    pts = np.asarray(c["data"]["points"][:3], dtype=float)  # (3, n_markers, n_frames)
    rate = float(params["RATE"]["value"][0])
    scale = 1.0
    units = params["UNITS"]["value"] if "UNITS" in params else []
    if to_meters and len(units) > 0 and str(units[0]).strip().lower() == "mm":
        scale = 1e-3
    markers = {}
    for i, lab in enumerate(labels):
        if lab and lab not in markers:
            markers[lab] = pts[:, i, :].T * scale  # (n_frames, 3)
    n_frames = pts.shape[2]
    return markers, rate, n_frames


def resolve_markers(needed, markers):
    """Map each needed name to its trajectory: exact match, else unique case-insensitive match."""
    view, missing, fallbacks = {}, set(), {}
    for name in needed:
        if name in markers:
            view[name] = markers[name]
            continue
        cands = [k for k in markers if k.lower() == name.lower()]
        if len(cands) == 1:
            view[name] = markers[cands[0]]
            fallbacks[name] = cands[0]
        else:
            missing.add(name)
    return view, missing, fallbacks


def process_trial(c3d_path, model, out_path):
    markers, rate, n = load_c3d(c3d_path, TO_METERS)
    needed = set().union(*(_markers_needed(d) for d in MODEL_DEFS[model].values()))
    view, _, fallbacks = resolve_markers(needed, markers)
    if fallbacks:
        print(f"  warning [{model}] {Path(c3d_path).name}: case-insensitive match used for {fallbacks}")
    data = {"frame": np.arange(n), "time": np.arange(n) / rate}
    n_ok, missing_all = 0, set()

    for seg, defn in MODEL_DEFS[model].items():
        cols_o = [f"{seg}_O_{k}" for k in "xyz"]
        cols_r = [f"{seg}_{k}" for k in R_NAMES]
        missing = _markers_needed(defn) - view.keys()
        if missing:
            missing_all |= missing
            origin, rot = np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan)
        else:
            origin, rot = compute_frame(defn, view)
            n_ok += 1
        data.update({c: origin[:, i] for i, c in enumerate(cols_o)})
        data.update({c: v for c, v in zip(cols_r, rot.reshape(n, 9).T)})

    if n_ok == 0:
        return "skipped (no segment computable)", missing_all
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(out_path, index=False)
    return ("ok" if not missing_all else "partial"), missing_all


def main(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT, models=MODELS, subjects=None, overwrite=False):
    files = sorted(Path(input_root).rglob("*.c3d"))
    if subjects:
        files = [f for f in files if f.relative_to(input_root).parts[0] in subjects]
    print(f"{len(files)} c3d files found in {input_root}")

    summary = {}
    for model in models:
        for f in files:
            rel = f.relative_to(input_root).with_suffix(".parquet")
            out = Path(output_root) / model / rel
            if out.exists() and not overwrite:
                status, missing = "exists (skipped)", set()
            else:
                try:
                    status, missing = process_trial(f, model, out)
                except Exception as e:  # keep going on a corrupted file
                    status, missing = f"ERROR {type(e).__name__}: {e}", set()
            summary[status.split(" ")[0]] = summary.get(status.split(" ")[0], 0) + 1
            msg = f"[{model}] {rel}: {status}"
            if missing:
                msg += f" | missing markers: {sorted(missing)}"
            print(msg)
    print("\nSummary:", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=INPUT_ROOT)
    p.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    main(a.input, a.output, a.models, a.subjects, a.overwrite)