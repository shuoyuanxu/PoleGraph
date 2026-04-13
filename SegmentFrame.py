"""
SegmentFrame.py
---------------
Instantaneous LiDAR frame → semantic pole representation.

Applies the same ground-map / trunk-slice / DBSCAN pipeline as Main.py but
operates on a single extracted PCD frame instead of the full aggregated map.

Usage:
    python3 SegmentFrame.py                                          # first ouster frame
    python3 SegmentFrame.py extracted_frames/velodyne/velodyne_frame0003_*.pcd
"""

import sys
import glob
import json
import random
import copy

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dataclasses import dataclass
from sklearn.cluster import DBSCAN


# ── Config ────────────────────────────────────────────────────────────────────
with open('config.json', 'r') as f:
    config = json.load(f)

MIN_POINTS_FOR_FIT = 5      # lower than map pipeline – single frames are sparser
PLOT_SCALE         = 3.0    # same as Main.py
SENSOR             = "ouster"
OVERLAY_ORIGINAL   = True  # True = grey raw point cloud background + coloured clusters

# Range-adaptive clustering
# Points beyond FAR_RANGE_M use relaxed DBSCAN params so sparse distant poles
# are still detected.  Set FAR_RANGE_M to a large value to disable.
FAR_RANGE_M        = 15.0   # metres XY – threshold between near and far zones
FAR_EPS_SCALE      = 2.5    # multiply config["eps"] for far-zone pass 1
FAR_MIN_SAMPLES    = 2      # min_samples for far-zone (very sparse returns)

# Vehicle / self-hit filter
# Points whose XY distance from the sensor origin is below this threshold are
# assumed to be returns off the vehicle body and are discarded before clustering.
VEHICLE_RADIUS_XY  = 1.5    # metres  – tune to your vehicle footprint
MAX_TILT_DEG       = 60.0   # discard poles tilted more than this (false positives)


# ── Data class (identical to Main.py) ─────────────────────────────────────────
@dataclass
class Pole:
    pole_id:     int
    centroid:    np.ndarray   # (x, y, z)
    radius:      float
    orientation: np.ndarray   # unit axis vector
    tilt_deg:    float
    n_points:    int


# ── Shared helpers (same logic as Main.py) ────────────────────────────────────
def random_color():
    return [random.random(), random.random(), random.random()]


def build_ground_map(points, grid_size, min_x, min_y):
    ground_map = {}
    for x, y, z in points:
        key = (int(np.floor((x - min_x) / grid_size)),
               int(np.floor((y - min_y) / grid_size)))
        ground_map[key] = min(ground_map.get(key, z), z)
    return ground_map


def extract_trunk_points(points, ground_map, min_x, min_y,
                         grid_size, ground_tol, trunk_min, trunk_max):
    trunk_mask = []
    for x, y, z in points:
        key = (int(np.floor((x - min_x) / grid_size)),
               int(np.floor((y - min_y) / grid_size)))
        if key in ground_map:
            h = z - ground_map[key]
            trunk_mask.append(trunk_min <= h <= trunk_max)
        else:
            trunk_mask.append(False)
    return np.array(trunk_mask)


def run_dbscan(pts, eps=None, min_samples=None):
    eps         = eps         or config["eps"]
    min_samples = min_samples or config["min_samples"]
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)


def filter_by_height_extent(pts, labels, trunk_min, trunk_max, min_ratio):
    min_extent = (trunk_max - trunk_min) * min_ratio
    keep = np.zeros(len(pts), dtype=bool)
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask   = labels == lbl
        z_range = pts[mask, 2].max() - pts[mask, 2].min()
        if z_range >= min_extent:
            keep[mask] = True
    return keep


def fit_pole(pole_id, cluster_pts):
    centroid = cluster_pts.mean(axis=0)
    centred  = cluster_pts - centroid

    cov              = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis             = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:
        axis = -axis

    vertical = np.array([0.0, 0.0, 1.0])
    tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(axis, vertical), -1, 1))))

    proj   = centred - np.outer(centred @ axis, axis)
    radius = float(np.linalg.norm(proj, axis=1).mean())

    return Pole(pole_id=pole_id, centroid=centroid, radius=radius,
                orientation=axis, tilt_deg=tilt_deg, n_points=len(cluster_pts))


def abstract_poles(clean_pts, labels):
    poles = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        cluster_pts = clean_pts[labels == lbl]
        if len(cluster_pts) < MIN_POINTS_FOR_FIT:
            continue
        poles.append(fit_pole(pole_id=lbl, cluster_pts=cluster_pts))
    return poles


# ── Vehicle / near-origin filter ──────────────────────────────────────────────
def remove_vehicle_points(points, radius_xy=VEHICLE_RADIUS_XY):
    """Remove points within radius_xy of (0,0) in the XY plane (vehicle body)."""
    xy_dist = np.linalg.norm(points[:, :2], axis=1)
    return points[xy_dist > radius_xy]


def filter_vehicle_poles(poles, radius_xy=VEHICLE_RADIUS_XY, max_tilt=MAX_TILT_DEG):
    """Drop any abstracted pole whose centroid is inside the vehicle radius or
    whose tilt exceeds max_tilt (near-horizontal → not a pole)."""
    kept = []
    for p in poles:
        dist = np.linalg.norm(p.centroid[:2])
        if dist <= radius_xy:
            print(f"  [vehicle filter] dropped pole {p.pole_id}: centroid {dist:.2f}m from origin")
            continue
        if p.tilt_deg > max_tilt:
            print(f"  [tilt filter]    dropped pole {p.pole_id}: tilt {p.tilt_deg:.1f}° > {max_tilt}°")
            continue
        kept.append(p)
    return kept


# ── Visualisation (same style as Main.py) ────────────────────────────────────
def visualise_poles_3d(poles, pole_pts, pole_labels, raw_points=None):
    geometries = []

    color_map_3d = {lbl: random_color() for lbl in np.unique(pole_labels)}

    # ── Optional grey background of the full raw point cloud ─────────────────
    if raw_points is not None:
        bg_pcd        = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(raw_points)
        bg_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(config["background_color"], (len(raw_points), 1))
        )
        geometries.append(bg_pcd)

    pcd_vis        = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pole_pts)
    pcd_vis.colors = o3d.utility.Vector3dVector(
        np.array([color_map_3d[l] for l in pole_labels])
    )
    geometries.append(pcd_vis)

    for p in poles:
        height   = config["trunk_height_max"] - config["trunk_height_min"]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=p.radius, height=height, resolution=20
        )

        z_axis = np.array([0.0, 0.0, 1.0])
        axis   = p.orientation
        v      = np.cross(z_axis, axis)
        s      = np.linalg.norm(v)
        c      = np.dot(z_axis, axis)

        if s < 1e-6:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R  = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        T         = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = p.centroid
        cylinder.transform(T)

        colour = color_map_3d.get(p.pole_id, [1.0, 0.0, 0.0])
        cylinder.paint_uniform_color(colour)
        cylinder.compute_vertex_normals()
        geometries.append(cylinder)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Pole Map vs Point Cloud",
                      width=config["window_size"]["width"],
                      height=config["window_size"]["height"])
    for g in geometries:
        vis.add_geometry(g)
    vis.get_render_option().point_size = config["point_size"]
    vis.run()
    vis.destroy_window()


def plot_pole_map_2d(poles):
    fig, ax = plt.subplots(figsize=(14, 12))

    for p in poles:
        cx, cy  = p.centroid[0], p.centroid[1]
        axis_xy = p.orientation[:2]
        xy_mag  = np.linalg.norm(axis_xy)
        minor   = p.radius * PLOT_SCALE * 2

        if xy_mag > 1e-3:
            cos_tilt  = max(abs(p.orientation[2]), 0.05)
            major     = (p.radius / cos_tilt) * PLOT_SCALE * 2
            angle_deg = np.degrees(np.arctan2(axis_xy[1], axis_xy[0]))
        else:
            major     = minor
            angle_deg = 0.0

        ax.add_patch(Ellipse(xy=(cx, cy), width=major, height=minor,
                             angle=angle_deg,
                             edgecolor='steelblue', facecolor='lightblue',
                             linewidth=0.8, alpha=0.7, zorder=2))

    if len(poles) <= 100:
        for p in poles:
            ax.text(p.centroid[0], p.centroid[1], str(p.pole_id),
                    ha='center', va='center', fontsize=5, zorder=3)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"Pole Map — {len(poles)} poles  |  ellipse = projected cylinder  |  "
                 f"scale ×{PLOT_SCALE}", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Core pipeline ─────────────────────────────────────────────────────────────
def segment_frame(pcd_path: str):
    print(f"\n[SegmentFrame] Loading: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  Raw points          : {len(pcd.points)}")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=config["nb_neighbors"],
                                             std_ratio=config["std_ratio"] * 4)
    points = np.asarray(pcd.points)
    print(f"  After denoising     : {len(points)}")

    # ── Vehicle / self-hit removal ────────────────────────────────────────────
    points = remove_vehicle_points(points, VEHICLE_RADIUS_XY)
    print(f"  After vehicle strip : {len(points)}  (r < {VEHICLE_RADIUS_XY}m removed)")

    if len(points) < 50:
        raise ValueError("Too few points after filtering – check the PCD file.")

    min_x, min_y = points[:, :2].min(axis=0)

    ground_map = build_ground_map(points, config["grid_size"], min_x, min_y)
    trunk_mask = extract_trunk_points(points, ground_map, min_x, min_y,
                                      config["grid_size"],
                                      config["ground_tolerance"],
                                      config["trunk_height_min"],
                                      config["trunk_height_max"])
    trunk_pts = points[trunk_mask]
    print(f"  Trunk-slice points  : {len(trunk_pts)}")

    if len(trunk_pts) < 10:
        print("  WARNING: very few trunk points.")
        return [], trunk_pts, np.array([]), points

    # ── Range-adaptive pass 1: DBSCAN → height-extent filter ────────────────────
    # Near zone: standard relaxed params; far zone: wider eps + lower min_samples
    eps_near  = config["eps"] * 1.5
    samp_near = max(3, config["min_samples"] // 3)
    eps_far   = config["eps"] * FAR_EPS_SCALE
    samp_far  = FAR_MIN_SAMPLES

    xy_dist_trunk = np.linalg.norm(trunk_pts[:, :2], axis=1)
    near_mask     = xy_dist_trunk <= FAR_RANGE_M
    far_mask      = ~near_mask
    print(f"  Trunk near/far split: {near_mask.sum()} near  /  {far_mask.sum()} far  (threshold {FAR_RANGE_M}m)")

    def _cluster_and_filter(pts):
        if len(pts) < 3:
            return np.zeros(0, dtype=bool)
        is_far = np.linalg.norm(pts[:, :2], axis=1) > FAR_RANGE_M
        labels = np.full(len(pts), -1, dtype=int)
        if (~is_far).any():
            labels[~is_far] = run_dbscan(pts[~is_far], eps=eps_near, min_samples=samp_near)
        if is_far.any():
            far_labels = run_dbscan(pts[is_far], eps=eps_far, min_samples=samp_far)
            # offset far cluster IDs to avoid collision with near IDs
            offset = int(labels.max()) + 1 if labels.max() >= 0 else 0
            labels[is_far] = np.where(far_labels == -1, -1, far_labels + offset)
        return filter_by_height_extent(pts, labels,
                                       config["trunk_height_min"],
                                       config["trunk_height_max"],
                                       config["min_height_ratio"])

    keep      = _cluster_and_filter(trunk_pts)
    clean_pts = trunk_pts[keep]
    print(f"  After height filter : {len(clean_pts)}")

    if len(clean_pts) < 5:
        print("  WARNING: no points survived height-extent filter.")
        return [], clean_pts, np.array([]), points

    # ── Pass 2: final DBSCAN on cleaned trunk points (same range split) ───────
    xy_dist_clean = np.linalg.norm(clean_pts[:, :2], axis=1)
    is_far2       = xy_dist_clean > FAR_RANGE_M
    labels2       = np.full(len(clean_pts), -1, dtype=int)
    if (~is_far2).any():
        labels2[~is_far2] = run_dbscan(clean_pts[~is_far2], eps=eps_near, min_samples=samp_near)
    if is_far2.any():
        far_labels2 = run_dbscan(clean_pts[is_far2], eps=eps_far, min_samples=samp_far)
        offset2     = int(labels2.max()) + 1 if labels2.max() >= 0 else 0
        labels2[is_far2] = np.where(far_labels2 == -1, -1, far_labels2 + offset2)

    n_clusters = len(set(labels2) - {-1})
    print(f"  Raw clusters        : {n_clusters}")

    poles_raw = abstract_poles(clean_pts, labels2)

    # ── Post-fit filters (vehicle centroid + tilt) ────────────────────────────
    poles = filter_vehicle_poles(poles_raw, VEHICLE_RADIUS_XY, MAX_TILT_DEG)
    print(f"  Poles after filter  : {len(poles)}")

    return poles, clean_pts, labels2, points


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        frame_path = sys.argv[1]
    else:
        matches = sorted(glob.glob(f"extracted_frames/{SENSOR}/{SENSOR}_frame0000_*.pcd"))
        if not matches:
            sys.exit(f"No frames found under extracted_frames/{SENSOR}/")
        frame_path = matches[0]

    poles, clean_pts, labels2, all_points = segment_frame(frame_path)

    # Keep only the clusters that survived all filters for 3-D viz
    surviving_ids = {p.pole_id for p in poles}
    if len(clean_pts) > 0 and len(labels2) > 0:
        vis_mask   = np.array([l in surviving_ids for l in labels2])
        vis_pts    = clean_pts[vis_mask]
        vis_labels = labels2[vis_mask]
    else:
        vis_pts    = clean_pts
        vis_labels = labels2

    # Print summary table
    print(f"\n{'ID':>4}  {'X':>8}  {'Y':>8}  {'Z':>8}  {'Radius':>8}  "
          f"{'Tilt°':>7}  {'Npts':>5}")
    print("-" * 60)
    for p in poles:
        print(f"{p.pole_id:>4}  {p.centroid[0]:>8.3f}  {p.centroid[1]:>8.3f}  "
              f"{p.centroid[2]:>8.3f}  {p.radius:>8.4f}  "
              f"{p.tilt_deg:>7.1f}  {p.n_points:>5}")

    plot_pole_map_2d(poles)

    if len(vis_pts) > 0:
        raw_bg = all_points if OVERLAY_ORIGINAL else None
        visualise_poles_3d(poles, vis_pts, vis_labels, raw_points=raw_bg)
