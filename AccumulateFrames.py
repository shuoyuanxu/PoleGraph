"""
AccumulateFrames.py
-------------------
Multi-frame LiDAR accumulation: ground removal → ICP scan matching → merge →
pole clustering (same pipeline as SegmentFrame.py).

Ground is stripped from each frame before ICP so the matcher only works on
structural points — fewer points, better-conditioned alignment, less storage.

Pipeline
--------
1. Load N consecutive frames and associate each with its nearest odometry pose.
2. Remove ground from every frame (height above per-cell min-z).
3. Downsample with a voxel grid (storage / speed).
4. ICP-align each frame into frame-0 coordinates using the odometry-derived
   relative transform as the initial guess.
5. Merge all aligned above-ground clouds into one accumulated cloud.
6. Trunk-slice → DBSCAN → height-extent filter → pole abstraction
   (identical logic to SegmentFrame.py).
7. Visualise with the same 3-D / 2-D layout as SegmentFrame.py.

Usage
-----
    python3 AccumulateFrames.py                   # first 10 ouster frames
    python3 AccumulateFrames.py ouster 10
    python3 AccumulateFrames.py velodyne 15
"""

import sys
import glob
import json
import csv
import random

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

sys.path.insert(0, '.')
from SegmentFrame import (
    Pole, random_color,
    build_ground_map, extract_trunk_points,
    filter_by_height_extent, fit_pole, abstract_poles,
    visualise_poles_3d, plot_pole_map_2d,
    MIN_POINTS_FOR_FIT, PLOT_SCALE, MAX_TILT_DEG,
)

# ── Config ────────────────────────────────────────────────────────────────────
with open('config.json', 'r') as f:
    config = json.load(f)

SENSOR           = "ouster"
N_FRAMES         = 10

# Voxel size used for downsampling before ICP and for the merged cloud.
# Smaller → more detail but slower ICP and larger memory.
VOXEL_SIZE       = 0.05   # metres

# ICP settings
ICP_MAX_DIST     = 0.3    # metres – max correspondence distance for refinement
ICP_ITERATIONS   = 50

# Accumulated-cloud DBSCAN — no range-adaptive zones after merging since the
# "sensor origin" is ambiguous in a multi-frame cloud.  A slightly relaxed eps
# absorbs minor ICP residuals without merging adjacent poles.
ACC_EPS_SCALE    = 1.5    # × config["eps"]
ACC_MIN_SAMPLES  = 5

OVERLAY_ORIGINAL = True   # show merged above-ground cloud as grey background


# ── Odometry helpers ──────────────────────────────────────────────────────────
def load_odometry(csv_path: str) -> list:
    """Return sorted list of (timestamp_s, T_4x4) from ROS odometry CSV."""
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_s = int(row['%time']) * 1e-9
            tx = float(row['field.pose.pose.position.x'])
            ty = float(row['field.pose.pose.position.y'])
            tz = float(row['field.pose.pose.position.z'])
            qx = float(row['field.pose.pose.orientation.x'])
            qy = float(row['field.pose.pose.orientation.y'])
            qz = float(row['field.pose.pose.orientation.z'])
            qw = float(row['field.pose.pose.orientation.w'])

            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            entries.append((ts_s, T))

    entries.sort(key=lambda e: e[0])
    print(f"  Loaded {len(entries)} odometry poses")
    return entries


def frame_timestamp_s(pcd_path: str) -> float:
    """Extract timestamp (seconds) from filename: sensor_frameNNNN_SEC_MS.pcd"""
    stem  = pcd_path.split('/')[-1].replace('.pcd', '')
    parts = stem.split('_')
    # last two parts are seconds and milliseconds
    return int(parts[-2]) + int(parts[-1]) * 1e-3


def nearest_pose(ts_s: float, odom: list) -> np.ndarray:
    """4×4 pose matrix whose timestamp is closest to ts_s."""
    timestamps = np.array([e[0] for e in odom])
    idx = int(np.argmin(np.abs(timestamps - ts_s)))
    return odom[idx][1]


# ── Ground removal ────────────────────────────────────────────────────────────
def remove_ground(points: np.ndarray) -> np.ndarray:
    """
    Keep all points that are more than ground_tolerance above the local
    ground surface (per grid-cell minimum z).  This is broader than the
    trunk slice — it retains trunks, canopy, and other structure needed
    by ICP to find good correspondences.
    """
    if len(points) < 10:
        return points

    min_x, min_y = points[:, :2].min(axis=0)
    ground_map   = build_ground_map(points, config["grid_size"], min_x, min_y)

    tol  = config["ground_tolerance"]
    gs   = config["grid_size"]
    keep = []
    for x, y, z in points:
        key = (int(np.floor((x - min_x) / gs)),
               int(np.floor((y - min_y) / gs)))
        gz  = ground_map.get(key, z)        # no cell → treat as above ground
        if z - gz > tol:
            keep.append([x, y, z])

    return np.array(keep, dtype=np.float64) if keep else np.zeros((0, 3))


# ── Accumulation ──────────────────────────────────────────────────────────────
def accumulate_frames(frame_paths: list, odom: list) -> np.ndarray:
    """
    Ground-remove each frame, ICP-align to frame-0 coordinates, merge.

    When odometry is available it provides the initial 6-DOF guess for ICP so
    the refinement only needs to close a small residual (≤ ICP_MAX_DIST).
    Without odometry the previous frame's ICP result is carried forward as a
    dead-reckoning estimate — less reliable over long sequences.

    Returns the merged above-ground cloud as an (N, 3) numpy array.
    """
    if not frame_paths:
        raise ValueError("No frame paths provided.")

    use_odom   = len(odom) > 0
    ts0        = frame_timestamp_s(frame_paths[0])
    T0         = nearest_pose(ts0, odom) if use_odom else np.eye(4)
    T0_inv     = np.linalg.inv(T0)

    accumulated_pcd = None       # grows frame by frame (voxel-downsampled)
    T_prev_icp      = np.eye(4)  # fallback dead-reckoning when no odometry

    for i, path in enumerate(frame_paths):
        fname = path.split('/')[-1]
        print(f"  [{i+1:2d}/{len(frame_paths)}] {fname}", end='  ', flush=True)

        # ── Load + statistical denoising ──────────────────────────────────────
        pcd = o3d.io.read_point_cloud(path)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=config["nb_neighbors"],
            std_ratio=config["std_ratio"] * 4,
        )
        pts = np.asarray(pcd.points)
        print(f"raw={len(pts):6d}", end='  ')

        # ── Ground removal ────────────────────────────────────────────────────
        pts_ag = remove_ground(pts)
        print(f"above-gnd={len(pts_ag):6d}", end='  ')

        if len(pts_ag) < 20:
            print("SKIP")
            continue

        # ── Voxel downsample ──────────────────────────────────────────────────
        src_pcd        = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(pts_ag)
        src_pcd        = src_pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"down={len(src_pcd.points):5d}", end='  ')

        # ── Frame 0: reference, no alignment needed ───────────────────────────
        if i == 0:
            accumulated_pcd = src_pcd
            print("[reference]")
            continue

        # ── Initial transform from odometry (or dead-reckoning) ───────────────
        if use_odom:
            ti     = frame_timestamp_s(path)
            Ti     = nearest_pose(ti, odom)
            T_init = T0_inv @ Ti
        else:
            T_init = T_prev_icp

        # ── ICP refinement ────────────────────────────────────────────────────
        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            accumulated_pcd,
            ICP_MAX_DIST,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=ICP_ITERATIONS
            ),
        )
        T_prev_icp = result.transformation
        print(f"ICP fitness={result.fitness:.3f}  rmse={result.inlier_rmse:.4f}", end='  ')

        # ── Merge ─────────────────────────────────────────────────────────────
        src_pcd.transform(result.transformation)
        accumulated_pcd = accumulated_pcd + src_pcd
        accumulated_pcd = accumulated_pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"merged={len(accumulated_pcd.points):6d}")

    if accumulated_pcd is None or len(accumulated_pcd.points) == 0:
        raise RuntimeError("Accumulation produced an empty cloud — check frame paths.")

    merged = np.asarray(accumulated_pcd.points)
    print(f"\n  Accumulated cloud   : {len(merged)} points")
    return merged


# ── Pole segmentation on merged cloud ─────────────────────────────────────────
def segment_merged(pts_merged: np.ndarray):
    """
    Trunk-slice → two-pass DBSCAN → height filter → pole abstraction.

    Identical logic to SegmentFrame.segment_frame but:
      • No vehicle-body radius filter (vehicle traversed multiple positions).
      • Flat DBSCAN params (no range-adaptive zones — sensor origin is
        ambiguous in a merged cloud).
    """
    min_x, min_y = pts_merged[:, :2].min(axis=0)
    ground_map   = build_ground_map(pts_merged, config["grid_size"], min_x, min_y)

    trunk_mask = extract_trunk_points(
        pts_merged, ground_map, min_x, min_y,
        config["grid_size"], config["ground_tolerance"],
        config["trunk_height_min"], config["trunk_height_max"],
    )
    trunk_pts = pts_merged[trunk_mask]
    print(f"  Trunk-slice points  : {len(trunk_pts)}")

    if len(trunk_pts) < 10:
        print("  WARNING: very few trunk points — check trunk_height_min/max in config.")
        return [], trunk_pts, np.array([]), pts_merged

    eps_acc  = config["eps"] * ACC_EPS_SCALE
    samp_acc = ACC_MIN_SAMPLES

    # Pass 1: DBSCAN + height-extent filter
    labels1   = DBSCAN(eps=eps_acc, min_samples=samp_acc).fit_predict(trunk_pts)
    keep      = filter_by_height_extent(
        trunk_pts, labels1,
        config["trunk_height_min"], config["trunk_height_max"],
        config["min_height_ratio"],
    )
    clean_pts = trunk_pts[keep]
    print(f"  After height filter : {len(clean_pts)}")

    if len(clean_pts) < 5:
        print("  WARNING: no points survived height-extent filter.")
        return [], clean_pts, np.array([]), pts_merged

    # Pass 2: final DBSCAN on cleaned points
    labels2    = DBSCAN(eps=eps_acc, min_samples=samp_acc).fit_predict(clean_pts)
    n_clusters = len(set(labels2) - {-1})
    print(f"  Raw clusters        : {n_clusters}")

    poles_raw = abstract_poles(clean_pts, labels2)

    # Tilt filter only (no vehicle-centroid filter)
    poles = [p for p in poles_raw if p.tilt_deg <= MAX_TILT_DEG]
    dropped = len(poles_raw) - len(poles)
    if dropped:
        print(f"  Tilt filter dropped : {dropped} cluster(s)")
    print(f"  Poles after filter  : {len(poles)}")

    return poles, clean_pts, labels2, pts_merged


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    sensor   = sys.argv[1] if len(sys.argv) > 1 else SENSOR
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else N_FRAMES

    print(f"\n{'='*70}")
    print(f"AccumulateFrames — ground removal → ICP merge → pole clustering")
    print(f"  Sensor   : {sensor}")
    print(f"  Frames   : {n_frames}")
    print(f"  Voxel    : {VOXEL_SIZE} m   ICP max dist : {ICP_MAX_DIST} m")
    print(f"  DBSCAN   : eps={config['eps'] * ACC_EPS_SCALE:.3f}  "
          f"min_samples={ACC_MIN_SAMPLES}")
    print(f"{'='*70}\n")

    # ── Frames ────────────────────────────────────────────────────────────────
    frame_files = sorted(glob.glob(f"extracted_frames/{sensor}/*.pcd"))[:n_frames]
    if not frame_files:
        sys.exit(f"No PCD files found under extracted_frames/{sensor}/")
    print(f"Found {len(frame_files)} frame(s)\n")

    # ── Odometry ──────────────────────────────────────────────────────────────
    odom = []
    try:
        odom = load_odometry("odometry.csv")
    except FileNotFoundError:
        print("  WARNING: odometry.csv not found — using ICP dead-reckoning only.\n"
              "  Results may degrade for frames with large inter-frame motion.\n")

    # ── Accumulate ────────────────────────────────────────────────────────────
    print("── Accumulating frames ─────────────────────────────────────────────")
    pts_merged = accumulate_frames(frame_files, odom)

    # ── Segment ───────────────────────────────────────────────────────────────
    print("\n── Segmenting merged cloud ──────────────────────────────────────────")
    poles, clean_pts, labels2, all_pts = segment_merged(pts_merged)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'ID':>4}  {'X':>8}  {'Y':>8}  {'Z':>8}  {'Radius':>8}  "
          f"{'Tilt°':>7}  {'Npts':>5}")
    print("-" * 60)
    for p in poles:
        print(f"{p.pole_id:>4}  {p.centroid[0]:>8.3f}  {p.centroid[1]:>8.3f}  "
              f"{p.centroid[2]:>8.3f}  {p.radius:>8.4f}  "
              f"{p.tilt_deg:>7.1f}  {p.n_points:>5}")

    # ── Visualise (same style as SegmentFrame) ────────────────────────────────
    plot_pole_map_2d(poles, raw_points=all_pts if OVERLAY_ORIGINAL else None)

    surviving_ids = {p.pole_id for p in poles}
    if len(clean_pts) > 0 and len(labels2) > 0:
        vis_mask   = np.array([l in surviving_ids for l in labels2])
        vis_pts    = clean_pts[vis_mask]
        vis_labels = labels2[vis_mask]
    else:
        vis_pts    = clean_pts
        vis_labels = labels2

    if len(vis_pts) > 0:
        raw_bg = all_pts if OVERLAY_ORIGINAL else None
        visualise_poles_3d(poles, vis_pts, vis_labels, raw_points=raw_bg)


if __name__ == "__main__":
    main()
