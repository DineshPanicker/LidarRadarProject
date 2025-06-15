import os
import json
import numpy as np
import open3d as o3d
from collections import defaultdict

# ========= CONFIGURATION ==========
SCENE_ID = "0000002033"
BASE_DIR = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply"

PLY_PATH = os.path.join(BASE_DIR, f"{SCENE_ID}_yolo_gt.ply")
JSON_PATH = os.path.join(BASE_DIR, f"{SCENE_ID}_gt_boxes.json")
ASSIGNMENT_PATH = os.path.join(BASE_DIR, f"{SCENE_ID}_assignment.npy")

# ========= FUNCTIONS ==========
def lighten_color(color, factor=0.7):
    # Blend color with white to lighten it
    white = np.array([1.0, 1.0, 1.0])
    color = np.array(color)
    light_color = color * factor + white * (1 - factor)
    light_color = np.clip(light_color, 0, 1)
    return light_color.tolist()

def create_bbox_lines(corners, color):
    lines = [[0,5],[1,4],[2,7],[3,6],
             [0,1],[1,3],[3,2],[2,0],
             [4,5],[5,7],[7,6],[6,4]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def points_in_obb(points, corners):
    center = np.mean(corners, axis=0)
    x_axis = corners[1] - corners[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = corners[3] - corners[1]
    y_axis /= np.linalg.norm(y_axis)
    z_axis = corners[4] - corners[0]
    z_axis /= np.linalg.norm(z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    relative_pts = points - center
    local_pts = relative_pts @ R

    extent_x = np.linalg.norm(corners[1] - corners[0]) / 2
    extent_y = np.linalg.norm(corners[3] - corners[1]) / 2
    extent_z = np.linalg.norm(corners[4] - corners[0]) / 2

    mask = (
        (np.abs(local_pts[:, 0]) <= extent_x) &
        (np.abs(local_pts[:, 1]) <= extent_y) &
        (np.abs(local_pts[:, 2]) <= extent_z)
    )
    return mask


def obb_to_aabb(corners):
    """Convert OBB corners to AABB box: returns min and max corners."""
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner


def aabb_intersection_volume(min1, max1, min2, max2):
    """Compute volume of intersection between two AABBs."""
    max_of_mins = np.maximum(min1, min2)
    min_of_maxs = np.minimum(max1, max2)
    delta = min_of_maxs - max_of_mins
    if np.any(delta <= 0):
        return 0.0
    return np.prod(delta)


def obb_volume(corners):
    """Compute volume of OBB from corners."""
    edge1 = np.linalg.norm(corners[1] - corners[0])
    edge2 = np.linalg.norm(corners[3] - corners[1])
    edge3 = np.linalg.norm(corners[4] - corners[0])
    return edge1 * edge2 * edge3


def obb_iou_3d(corners1, corners2, iou_threshold=0.3):
    """
    Approximate 3D IoU between two oriented bounding boxes by using
    their axis-aligned bounding boxes (AABB).
    """
    min1, max1 = obb_to_aabb(corners1)
    min2, max2 = obb_to_aabb(corners2)

    inter_vol = aabb_intersection_volume(min1, max1, min2, max2)
    if inter_vol == 0:
        return 0.0

    vol1 = obb_volume(corners1)
    vol2 = obb_volume(corners2)
    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / union_vol
    return iou


# ========= MAIN ==========

pcd = o3d.io.read_point_cloud(PLY_PATH)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
assignment = np.load(ASSIGNMENT_PATH)

with open(JSON_PATH, 'r') as f:
    boxes = json.load(f)

# === Group boxes by color ===
boxes_by_color = defaultdict(list)

for box in boxes:
    corners_np = np.array(box)
    inside_mask = points_in_obb(points, corners_np)
    box_assignments = assignment[inside_mask]
    box_assignments = box_assignments[box_assignments >= 0]

    if len(box_assignments) > 0:
        car_id = np.bincount(box_assignments).argmax()
        car_color = colors[assignment == car_id][0] if np.any(assignment == car_id) else [0, 0, 0]
        volume = obb_volume(corners_np)
        points_inside = np.sum(inside_mask)
        density = points_inside / volume if volume > 0 else 0

        box_info = {
            'corners': corners_np,
            'color': car_color.tolist(),
            'num_points': points_inside,
            'volume': volume,
            'density': density
        }

        boxes_by_color[tuple(car_color)].append(box_info)

# === Remove overlapping boxes using IoU ===
final_boxes = []
iou_thresh = 0.3  # Adjust IoU threshold here

for color, box_list in boxes_by_color.items():
    selected = []
    box_list = sorted(box_list, key=lambda b: b['num_points'], reverse=True)

    for box_info in box_list:
        overlaps = False
        for sel in selected:
            iou = obb_iou_3d(box_info['corners'], sel['corners'], iou_threshold=iou_thresh)
            if iou > iou_thresh:
                overlaps = True
                break
        if not overlaps:
            selected.append(box_info)

    final_boxes.extend(selected)
# # === Add unmatched ground truth boxes (not matched to any detection) ===
# used_corners = [tuple(b['corners'].flatten()) for b in final_boxes]
#
# for box in boxes:
#     corners_np = np.array(box)
#     flat = tuple(corners_np.flatten())
#
#     if flat not in used_corners:
#         # Use red for unmatched boxes
#         red_color = [1.0, 0.0, 0.0]
#         box_info = {
#             'corners': corners_np,
#             'color': red_color
#         }
#         final_boxes.append(box_info)

# === Draw final boxes ===
# === Draw final boxes with lighter lines ===
colored_boxes = [create_bbox_lines(b['corners'], lighten_color(b['color'], factor=0.7)) for b in final_boxes]
o3d.visualization.draw_geometries([pcd] + colored_boxes)