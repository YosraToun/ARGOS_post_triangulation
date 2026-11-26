from pathlib import Path
import shutil
import matplotlib

# Backend non-interactif adapté à VS Code sans fenêtre graphique interactive
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


PARQUET_ROOT = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\reference_frames"
)
VIDEO_ROOT = Path(
    r"D:\Travail_ARGOS_Yosra\workspace\ARGOS_post_triangulation\reference_frame_videos"
)

START_SUBJECT = "S08"
MODELS = ("marker_based", "synthrtm", "synthpose", "rtmpose", "fake_rtmpose")
PREFER_MP4 = False  # Mets True uniquement si ton ffmpeg fonctionne sans erreur 0xC0000139


def compute_global_bounds(df: pl.DataFrame, margin: float = 0.15) -> tuple[tuple[float, float], ...]:
    """Calcule des bornes X, Y, Z fixes sur tout l'essai pour éviter que la caméra ne saute."""
    ox = df["origin_x"].to_numpy()
    oy = df["origin_y"].to_numpy()
    oz = df["origin_z"].to_numpy()

    valid = np.isfinite(ox) & np.isfinite(oy) & np.isfinite(oz)
    if not np.any(valid):
        return ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))

    ox, oy, oz = ox[valid], oy[valid], oz[valid]

    mins = np.array([ox.min(), oy.min(), oz.min()])
    maxs = np.array([ox.max(), oy.max(), oz.max()])
    centers = 0.5 * (mins + maxs)
    half_span = 0.5 * max(np.max(maxs - mins), 0.15) + margin

    return (
        (centers[0] - half_span, centers[0] + half_span),
        (centers[1] - half_span, centers[1] + half_span),
        (centers[2] - half_span, centers[2] + half_span),
    )


def animate_trial_frames(
    df: pl.DataFrame,
    out_path: Path,
    frame_step: int = 2,
    axis_scale: float = 0.08,
    fps: int = 20,
    dpi: int = 120,
    prefer_mp4: bool = PREFER_MP4,
) -> None:
    """Crée et sauvegarde une animation 3D avec TOUS les segments présents dans le Parquet."""
    # Récupère absolument tous les segments existants dans le fichier Parquet
    segments = [s for s in df["segment"].unique().sort().to_list() if s is not None]
    if not segments:
        print(f"  [WARN] Aucun segment trouvé dans {out_path}")
        return

    all_frames = sorted(df["frame"].unique().to_list())[::frame_step]
    if not all_frames:
        return

    cmap = plt.get_cmap("tab10")
    seg_colors = {seg: cmap(i % 10) for i, seg in enumerate(segments)}

    seg_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for seg in segments:
        sub = df.filter(pl.col("segment") == seg).sort("frame")
        f_arr = sub["frame"].to_numpy()
        o_arr = sub.select(["origin_x", "origin_y", "origin_z"]).to_numpy()
        ex_arr = sub.select(["ex_x", "ex_y", "ex_z"]).to_numpy()
        ey_arr = sub.select(["ey_x", "ey_y", "ey_z"]).to_numpy()
        ez_arr = sub.select(["ez_x", "ez_y", "ez_z"]).to_numpy()

        seg_data[seg] = {
            int(fr): (o_arr[i], ex_arr[i], ey_arr[i], ez_arr[i])
            for i, fr in enumerate(f_arr)
        }

    xlim, ylim, zlim = compute_global_bounds(df)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    model_name = df["model"][0] if len(df) else ""
    subj_name = df["subject"][0] if len(df) else ""
    trial_name = df["trial"][0] if len(df) else ""

    def update(frame_idx: int):
        fr = all_frames[frame_idx]
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"{model_name} | {subj_name} | {trial_name} — Frame {fr}\n"
            f"Segments ({len(segments)}): {', '.join(segments)}  |  Vecteurs: R=ex, G=ey, B=ez"
        )

        origins_cur = []
        for seg in segments:
            item = seg_data[seg].get(fr)
            if item is None:
                continue
            o, ex, ey, ez = item
            if not (np.all(np.isfinite(o)) and np.all(np.isfinite(ex))):
                continue

            col_seg = seg_colors[seg]
            origins_cur.append(o)
            ax.scatter(o[0], o[1], o[2], s=35, color=col_seg, edgecolors="black", linewidth=0.5)
            ax.text(o[0], o[1], o[2], f" {seg}", fontsize=7.5, color=col_seg, fontweight="bold")

            for vec, col_axis in [(ex, "tab:red"), (ey, "tab:green"), (ez, "tab:blue")]:
                nrm = np.linalg.norm(vec)
                if nrm > 1e-8:
                    u = vec / nrm
                    ax.quiver(
                        o[0], o[1], o[2],
                        u[0], u[1], u[2],
                        length=axis_scale,
                        color=col_axis,
                        linewidth=1.6,
                    )

        if len(origins_cur) > 1:
            pts = np.array(origins_cur)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="gray", alpha=0.4, linewidth=1.0)

        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(all_frames),
        interval=1000 / fps,
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved_mp4 = False
    if prefer_mp4 and shutil.which("ffmpeg") is not None:
        try:
            mp4_path = out_path.with_suffix(".mp4")
            anim.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000), dpi=dpi)
            saved_mp4 = True
        except Exception:
            saved_mp4 = False

    if not saved_mp4:
        gif_path = out_path.with_suffix(".gif")
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)


def run_all_videos(
    parquet_root: Path = PARQUET_ROOT,
    video_root: Path = VIDEO_ROOT,
    min_subject: str = START_SUBJECT,
    models: tuple[str, ...] = MODELS,
    frame_step: int = 3,
    axis_scale: float = 0.08,
    fps: int = 20,
    overwrite: bool = True,
) -> None:
    count = 0
    for model in models:
        model_dir = parquet_root / model
        if not model_dir.exists():
            continue

        subj_dirs = sorted(
            [p for p in model_dir.iterdir() if p.is_dir() and p.name >= min_subject],
            key=lambda p: p.name,
        )

        for subj_dir in subj_dirs:
            subject_id = subj_dir.name
            for pq_path in sorted(subj_dir.rglob("*.parquet")):
                trial_rel = pq_path.relative_to(subj_dir)
                base_out = video_root / model / subject_id / trial_rel.with_suffix("")

                if not overwrite and (
                    base_out.with_suffix(".mp4").exists() or base_out.with_suffix(".gif").exists()
                ):
                    continue

                df_trial = pl.read_parquet(pq_path)
                if len(df_trial) == 0:
                    continue

                present_segments = sorted(df_trial["segment"].unique().to_list())
                count += 1
                print(f"[{count}] {model} | {subject_id} | {trial_rel} -> segments: {present_segments}")
                try:
                    animate_trial_frames(
                        df=df_trial,
                        out_path=base_out,
                        frame_step=frame_step,
                        axis_scale=axis_scale,
                        fps=fps,
                    )
                except Exception as exc:
                    print(f"  [ERREUR] {pq_path}: {exc}")


if __name__ == "__main__":
    run_all_videos(
        parquet_root=PARQUET_ROOT,
        video_root=VIDEO_ROOT,
        min_subject=START_SUBJECT,
        models=MODELS,
        frame_step=3,
        axis_scale=0.08,
        fps=20,
        overwrite=True,
    )