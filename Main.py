import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
import copy
import json
import csv
from dataclasses import dataclass
from plyfile import PlyData, PlyElement


@dataclass
class Pole:
    pole_id:     int
    centroid:    np.ndarray   # (x, y, z)
    radius:      float        # metres
    orientation: np.ndarray   # unit axis vector
    tilt_deg:    float        # degrees from vertical
    n_points:    int


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


def run_dbscan(pts):
    labels = DBSCAN(eps=config["eps"],
                    min_samples=config["min_samples"]).fit_predict(pts)
    return labels


def filter_by_height_extent(pts, labels, trunk_min, trunk_max, min_ratio):
    min_extent = (trunk_max - trunk_min) * min_ratio
    keep = np.zeros(len(pts), dtype=bool)
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
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


def save_pole_map_csv(poles, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'diameter',
                         'orientation_x', 'orientation_y', 'orientation_z'])
        for p in poles:
            writer.writerow([
                p.pole_id,
                round(p.centroid[0], 4),
                round(p.centroid[1], 4),
                round(p.radius * 2, 4),
                round(p.orientation[0], 4),
                round(p.orientation[1], 4),
                round(p.orientation[2], 4),
            ])
    print(f"Pole map saved → {path}  ({len(poles)} poles)")

def visualise_poles_3d(poles, pole_pts, pole_labels):
    geometries = []

    # ── Pole point cloud (coloured by cluster) ────────────────────────────────
    color_map_3d = {lbl: random_color() for lbl in np.unique(pole_labels)}
    pcd_vis        = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pole_pts)
    pcd_vis.colors = o3d.utility.Vector3dVector(
        np.array([color_map_3d[l] for l in pole_labels])
    )
    geometries.append(pcd_vis)

    # ── Fitted cylinders from pole abstraction ────────────────────────────────
    for p in poles:
        # open3d cylinder is created along Z axis, then rotated to pole orientation
        height   = config["trunk_height_max"] - config["trunk_height_min"]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=p.radius, height=height, resolution=20
        )

        # Rotation: align Z axis [0,0,1] to pole orientation axis
        z_axis  = np.array([0.0, 0.0, 1.0])
        axis    = p.orientation
        v       = np.cross(z_axis, axis)
        s       = np.linalg.norm(v)
        c       = np.dot(z_axis, axis)

        if s < 1e-6:
            # already aligned or anti-aligned
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R  = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        T           = np.eye(4)
        T[:3, :3]   = R
        T[:3,  3]   = p.centroid  # translate to pole centroid
        cylinder.transform(T)

        # Wire-frame look: paint cylinder with cluster colour, semi-transparent
        colour = color_map_3d.get(p.pole_id, [1.0, 0.0, 0.0])
        cylinder.paint_uniform_color(colour)
        cylinder.compute_vertex_normals()
        geometries.append(cylinder)

    # ── Coordinate frame at origin ────────────────────────────────────────────
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


# ============================================================================
# LOAD CONFIG
# ============================================================================
with open('config.json', 'r') as f:
    config = json.load(f)

OVERLAY_ORIGINAL   = False  # True = grey background + clusters | False = clusters only
SAVE_POLE_PCD      = True   # save clustered poles as PLY for Graphbuilding.py
POLE_PCD_PATH      = "./poles_clustered.ply"
POLE_CSV_PATH      = "./pole_map.csv"

# Pole abstraction
MIN_POINTS_FOR_FIT = 10     # minimum cluster points to attempt pole fit
PLOT_SCALE         = 3.0    # scale ellipse size for visibility (1.0 = true size)
# ============================================================================

# ── Load & denoise ───────────────────────────────────────────────────────────
pcd = o3d.io.read_point_cloud("./SurfMap.pcd")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=config["nb_neighbors"],
                                         std_ratio=config["std_ratio"])
points = np.asarray(pcd.points)
min_x, min_y = points[:, :2].min(axis=0)

# ── Ground map & trunk slice ─────────────────────────────────────────────────
ground_map = build_ground_map(points, config["grid_size"], min_x, min_y)
trunk_mask = extract_trunk_points(points, ground_map, min_x, min_y,
                                  config["grid_size"],
                                  config["ground_tolerance"],
                                  config["trunk_height_min"],
                                  config["trunk_height_max"])
trunk_pts  = points[trunk_mask]

# ── Pass 1: DBSCAN → filter by vertical extent ───────────────────────────────
labels1   = run_dbscan(trunk_pts)
keep      = filter_by_height_extent(trunk_pts, labels1,
                                    config["trunk_height_min"],
                                    config["trunk_height_max"],
                                    config["min_height_ratio"])
clean_pts = trunk_pts[keep]

# ── Pass 2: DBSCAN on cleaned points ─────────────────────────────────────────
labels2    = run_dbscan(clean_pts)
n_clusters = len(set(labels2) - {-1})
print(f"Final clusters (poles): {n_clusters}")

# ── Pole abstraction → CSV + 2D map ──────────────────────────────────────────
poles = abstract_poles(clean_pts, labels2)
save_pole_map_csv(poles, POLE_CSV_PATH)
plot_pole_map_2d(poles)

cluster_mask_vis = labels2 != -1
visualise_poles_3d(poles, clean_pts[cluster_mask_vis], labels2[cluster_mask_vis])
# ── Save clustered poles as PLY ───────────────────────────────────────────────
if SAVE_POLE_PCD:
    cluster_mask   = labels2 != -1
    pole_pts       = clean_pts[cluster_mask]
    pole_labels    = labels2[cluster_mask]
    color_map_save = {lbl: [int(c * 255) for c in random_color()]
                      for lbl in np.unique(pole_labels)}

    vertex_data = np.array(
        [(p[0], p[1], p[2],
          color_map_save[l][0],
          color_map_save[l][1],
          color_map_save[l][2],
          int(l))
         for p, l in zip(pole_pts, pole_labels)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('label', 'i4')]
    )
    PlyData([PlyElement.describe(vertex_data, 'vertex')],
            text=True).write(POLE_PCD_PATH)
    print(f"Saved {len(pole_pts)} pole points → {POLE_PCD_PATH}")

# ── Visualise point cloud ─────────────────────────────────────────────────────
color_map    = {lbl: random_color() for lbl in set(labels2) if lbl != -1}
orig_indices = np.where(trunk_mask)[0][keep]

if OVERLAY_ORIGINAL:
    vis_colors = np.tile(config["background_color"], (len(points), 1))
    for i, orig_idx in enumerate(orig_indices):
        lbl = labels2[i]
        if lbl != -1:
            vis_colors[orig_idx] = color_map[lbl]
    result_pcd        = copy.deepcopy(pcd)
    result_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
else:
    cluster_mask      = labels2 != -1
    result_pcd        = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(clean_pts[cluster_mask])
    result_pcd.colors = o3d.utility.Vector3dVector(
        np.array([color_map[lbl] for lbl in labels2[cluster_mask]])
    )

vis = o3d.visualization.Visualizer()
vis.create_window("Detected Pole Clusters",
                  width=config["window_size"]["width"],
                  height=config["window_size"]["height"])
vis.add_geometry(result_pcd)
vis.get_render_option().point_size = config["point_size"]
vis.run()
vis.destroy_window()