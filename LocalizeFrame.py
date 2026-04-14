"""
LocalizeFrame.py - Simplified
Search for best-matching region in global map for each scan frame.
Uses relative pole topology (distances between pairs) rather than absolute coordinates.
"""

import sys
import glob
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.spatial.distance import cdist
from dataclasses import dataclass

# Import from SegmentFrame
sys.path.insert(0, '.')
from SegmentFrame import segment_frame, Pole, random_color, PLOT_SCALE

with open('config.json', 'r') as f:
    config = json.load(f)

SENSOR = "ouster"
MAX_FRAMES = 3
MAX_MATCH_DIST = 3.0
INTERACTIVE = True  # Set to True to manually choose initial pose on map


@dataclass
class GlobalPole:
    pole_id: int
    centroid: np.ndarray


def load_global_poles(csv_path: str):
    """Load global pole map from CSV."""
    poles = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = float(row['x']), float(row['y'])
            poles.append(GlobalPole(
                pole_id=int(row['id']),
                centroid=np.array([x, y])
            ))
    print(f"Loaded {len(poles)} global poles")
    return poles


def compute_pairwise_distances(poles):
    """Return NxN distance matrix and list of (i,j) pairs."""
    n = len(poles)
    if n < 2:
        return np.array([]), []

    pts = np.array([p.centroid[:2] for p in poles])
    dist = cdist(pts, pts, metric='euclidean')

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, dist[i, j]))

    return dist, pairs


def match_topology(scan_poles, global_poles, seed_pos=None):
    """Find the scan's location by matching pole topology.

    seed_pos: (x, y, theta_rad) tuple if provided by the user, else None.
    When a seed is given the search is restricted to poles within 5 m of the
    seed position, and the seed theta is used as the initial rotation.

    Returns: best_transform (tx, ty, theta_rad), n_matches, quality_score
    """
    if len(scan_poles) < 2:
        return (0, 0, 0), 0, 0

    _, scan_pairs = compute_pairwise_distances(scan_poles)
    _, global_pairs = compute_pairwise_distances(global_poles)

    # Unpack seed – support both (x, y) and (x, y, theta) for robustness
    seed_theta = 0.0
    if seed_pos is not None:
        if len(seed_pos) == 3:
            seed_x, seed_y, seed_theta = seed_pos
        else:
            seed_x, seed_y = seed_pos
        search_poles = [gp for gp in global_poles
                        if np.linalg.norm(gp.centroid - np.array([seed_x, seed_y])) < 5.0]
    else:
        search_poles = global_poles

    best_score = -np.inf
    best_transform = (0, 0, seed_theta)
    best_matches = 0

    for anchor_gi in range(len(search_poles)):
        offset = search_poles[anchor_gi].centroid - scan_poles[0].centroid[:2]

        matches = 0
        total_pairs = len(scan_pairs)

        for i, j, dist_ij in scan_pairs:
            found = False
            for gi, gj, gdist_ij in global_pairs:
                if abs(gdist_ij - dist_ij) < 0.5:
                    found = True
                    break
            if found:
                matches += 1

        score = matches / (total_pairs + 1e-6)

        if score > best_score:
            best_score = score
            best_transform = (offset[0], offset[1], seed_theta)
            best_matches = matches

    return best_transform, best_matches, best_score


def get_interactive_seed(global_poles):
    """Let user click on map to set initial robot pose (position + heading).

    Interaction:
      - 1st left-click  : sets robot position (x, y)
      - 2nd left-click  : sets heading direction (angle from 1st to 2nd click)
      - Right-click / Escape : cancel at any stage

    Returns: (x, y, theta_rad) or None if cancelled
    """
    state = {
        'phase': 'position',   # 'position' -> 'heading' -> 'done'
        'pos': None,           # (x, y)
        'theta': None,         # radians
        'cancelled': False,
    }

    pos_marker = [None]
    heading_arrow = [None]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot global poles
    for gp in global_poles:
        ax.plot(gp.centroid[0], gp.centroid[1], 'o', color='#cccccc',
                markersize=8, zorder=1, alpha=0.6)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Step 1/2 – Left-click to place robot position\n"
                 "(right-click or Esc to cancel)",
                 fontsize=12, fontweight='bold')

    def refresh():
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'escape':
            state['cancelled'] = True
            plt.close()

    def on_click(event):
        if event.button == 3:          # right-click = cancel
            state['cancelled'] = True
            plt.close()
            return

        if event.button != 1:          # ignore middle button etc.
            return

        if event.inaxes is None or event.xdata is None:
            return

        if state['phase'] == 'position':
            x, y = event.xdata, event.ydata
            state['pos'] = (x, y)
            state['phase'] = 'heading'

            # Draw position marker
            marker, = ax.plot(x, y, 'r+', markersize=16,
                              markeredgewidth=2.5, zorder=5)
            pos_marker[0] = marker

            ax.set_title("Step 2/2 – Left-click to set heading direction\n"
                         f"Position locked: ({x:+.2f} m, {y:+.2f} m)  |  "
                         "right-click or Esc to cancel",
                         fontsize=12, fontweight='bold')
            refresh()

        elif state['phase'] == 'heading':
            x0, y0 = state['pos']
            x1, y1 = event.xdata, event.ydata
            theta = np.arctan2(y1 - y0, x1 - x0)
            state['theta'] = theta
            state['phase'] = 'done'

            # Remove previous arrow if re-clicked
            if heading_arrow[0] is not None:
                heading_arrow[0].remove()

            arrow_len = max(3.0,
                            0.08 * max(ax.get_xlim()[1] - ax.get_xlim()[0],
                                       ax.get_ylim()[1] - ax.get_ylim()[0]))
            end = (x0 + arrow_len * np.cos(theta),
                   y0 + arrow_len * np.sin(theta))
            arr = FancyArrowPatch(
                (x0, y0), end,
                arrowstyle='->', mutation_scale=25,
                color='red', lw=2.5, zorder=4
            )
            ax.add_patch(arr)
            heading_arrow[0] = arr

            ax.set_title(
                f"Initial pose set – ({x0:+.2f} m, {y0:+.2f} m, "
                f"{np.degrees(theta):+.1f}°)\n"
                "Close window or press Enter to confirm",
                fontsize=12, fontweight='bold', color='darkgreen'
            )
            refresh()

    def on_key_confirm(event):
        if event.key == 'enter' and state['phase'] == 'done':
            plt.close()
        elif event.key == 'escape':
            state['cancelled'] = True
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_confirm)
    plt.tight_layout()
    plt.show()

    if state['cancelled'] or state['pos'] is None or state['theta'] is None:
        return None

    x, y = state['pos']
    return (x, y, state['theta'])


def visualize_result(scan_poles, global_poles, tx, ty, theta, quality, frame_label):
    """Plot scan poles vs global poles with estimated pose (x, y, theta)."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot global poles (grey background)
    for gp in global_poles:
        ax.plot(gp.centroid[0], gp.centroid[1], 'o', color='#cccccc',
                markersize=6, zorder=1, alpha=0.7)

    # Rotation matrix for yaw
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Plot scan poles (transformed to estimated global frame)
    colors = {}
    for i, sp in enumerate(scan_poles):
        col = random_color()
        colors[i] = col
        # Transform: rotate then translate
        local_pos = sp.centroid[:2]
        rotated = R @ local_pos
        global_pos = rotated + np.array([tx, ty])
        ax.plot(global_pos[0], global_pos[1], 's', color=col, markersize=10,
                zorder=3, markeredgecolor='black', markeredgewidth=1)

    # Draw ellipses for visualized poles
    for i, sp in enumerate(scan_poles):
        local_pos = sp.centroid[:2]
        rotated = R @ local_pos
        global_pos = rotated + np.array([tx, ty])

        axis_xy = sp.orientation[:2]
        xy_mag = np.linalg.norm(axis_xy)
        minor = sp.radius * PLOT_SCALE * 2

        if xy_mag > 1e-3:
            cos_tilt = max(abs(sp.orientation[2]), 0.05)
            major = (sp.radius / cos_tilt) * PLOT_SCALE * 2
            angle_deg = np.degrees(np.arctan2(axis_xy[1], axis_xy[0]))
        else:
            major = minor
            angle_deg = 0.0

        # Rotate the ellipse angle by theta
        rotated_angle = angle_deg + np.degrees(theta)

        ax.add_patch(Ellipse(xy=global_pos, width=major, height=minor,
                             angle=rotated_angle,
                             edgecolor=colors[i], facecolor='none',
                             linewidth=1.5, alpha=0.8, zorder=2))

    # Mark estimated position and orientation
    # Position: origin point
    ax.plot(0, 0, 'r*', markersize=15, label='Scan origin', zorder=5)

    # Est. position with orientation arrow
    arrow_len = 2.0
    arrow_end = np.array([tx, ty]) + arrow_len * np.array([cos_t, sin_t])
    arrow = FancyArrowPatch((tx, ty), tuple(arrow_end),
                            arrowstyle='->', mutation_scale=25,
                            color='red', lw=2.5, zorder=4)
    ax.add_patch(arrow)
    ax.plot(tx, ty, 'r+', markersize=14, markeredgewidth=2.5, zorder=5, label='Est. pose')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"Localization: {frame_label}\n"
                 f"Quality: {quality:.1%} | Pose: ({tx:+.2f}m, {ty:+.2f}m, {np.degrees(theta):+.1f}°)",
                 fontsize=12)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()


def main():
    sensor = sys.argv[1] if len(sys.argv) > 1 else SENSOR
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_FRAMES

    print(f"\n{'='*70}")
    print(f"LocalizeFrame: Graph topology matching")
    print(f"{'='*70}\n")

    global_poles = load_global_poles("pole_map.csv")

    # Get frames
    frame_files = sorted(glob.glob(f"extracted_frames/{sensor}/*.pcd"))[:max_frames]
    print(f"Processing {len(frame_files)} frames\n")

    for frame_idx, frame_path in enumerate(frame_files):
        label = frame_path.split("/")[-1]
        print(f"[Frame {frame_idx}] {label}")

        try:
            scan_poles, _, _, _ = segment_frame(frame_path)

            if len(scan_poles) < 2:
                print(f"  → Insufficient poles ({len(scan_poles)})\n")
                continue

            # Interactive seed selection (optional)
            seed_pos = None
            if INTERACTIVE:
                print("  → Click on map to set initial pose (position + heading)...")
                seed_pos = get_interactive_seed(global_poles)
                if seed_pos is None:
                    print("  → Cancelled\n")
                    continue
                sx, sy, stheta = seed_pos
                print(f"  → Seed pose: ({sx:+.2f} m, {sy:+.2f} m, {np.degrees(stheta):+.1f}°)")

            # Find best match in global map
            (tx, ty, theta), n_matches, quality = match_topology(scan_poles, global_poles, seed_pos=seed_pos)

            print(f"  Poles: {len(scan_poles)}")
            print(f"  Matches: {n_matches}/{len(compute_pairwise_distances(scan_poles)[1])}")
            print(f"  Quality: {quality:.2%}")
            print(f"  Est. pose: x={tx:+.2f}m, y={ty:+.2f}m, θ={np.degrees(theta):+.1f}°\n")

            # Visualize result
            visualize_result(scan_poles, global_poles, tx, ty, theta, quality, label)

        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == "__main__":
    main()
