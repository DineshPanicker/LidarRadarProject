import os
import numpy as np
import json
import open3d as o3d
import pandas as pd

# scene_id = "0000002903"
base_dir = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply"
# ply_path = os.path.join(base_dir, f"{scene_id}_yolo_gt.ply")
# json_path = os.path.join(base_dir, f"{scene_id}_gt_boxes.json")
# assignment_path = os.path.join(base_dir, f"{scene_id}_assignment.npy")

# Loop through all scene files
for file in os.listdir(base_dir):
    if file.endswith("_yolo_gt.ply"):
        scene_id = file.replace("_yolo_gt.ply", "")
        ply_path = os.path.join(base_dir, f"{scene_id}_yolo_gt.ply")
        json_path = os.path.join(base_dir, f"{scene_id}_gt_boxes.json")
        assignment_path = os.path.join(base_dir, f"{scene_id}_assignment.npy")

        # Load data
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        assignment = np.load(assignment_path)
        with open(json_path, "r") as f:
            gt_boxes = json.load(f)

        # Check inside OBB
        def points_in_obb(points, corners):
            center = np.mean(corners, axis=0)
            x_axis = corners[1] - corners[0]
            y_axis = corners[3] - corners[1]
            z_axis = corners[4] - corners[0]
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis /= np.linalg.norm(z_axis)
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            local = (points - center) @ R
            ex = np.linalg.norm(corners[1] - corners[0]) / 2
            ey = np.linalg.norm(corners[3] - corners[1]) / 2
            ez = np.linalg.norm(corners[4] - corners[0]) / 2
            return (
                (np.abs(local[:, 0]) <= ex) &
                (np.abs(local[:, 1]) <= ey) &
                (np.abs(local[:, 2]) <= ez)
            )

        # Evaluation
        results = []
        for car_id in np.unique(assignment):
            if car_id == -1:
                continue
            mask_points = points[assignment == car_id]
            num_inside_total = 0
            for box in gt_boxes:
                corners = np.array(box)
                inside_mask = points_in_obb(mask_points, corners)
                num_inside = np.sum(inside_mask)
                num_inside_total += num_inside

            if num_inside_total == 0:
                continue  # Skip cars with 0 points inside the bbox

            total_mask = len(mask_points)
            bleed_out = total_mask - num_inside_total
            percentage_inside = (num_inside_total / total_mask) * 100 if total_mask > 0 else 0

            results.append({
                "car_id": int(car_id),
                "mask_points": int(total_mask),
                "inside_points": int(num_inside_total),
                "bleed_out": int(bleed_out),
                "percentage_inside": float(percentage_inside)
            })

        # Print scene ID
        print(f"Results for Scene ID: {scene_id}\n")

        df = pd.DataFrame(results)
        print(df.to_string(index=False))