import numpy as np
import open3d as o3d
import cv2
import torch
from ultralytics import YOLO

# ==== FILE PATHS ====
YOLO_MODEL_PATH = r"A:\RWU\Second Sem\Lidar and Radar\yolov8s-seg.pt"
LIDAR_BIN_PATH = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project\data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data\0000000100.bin"
IMAGE_PATH = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project\data_2d_raw\2013_05_28_drive_0000_sync\image_00\data_rect\0000000100.png"
PLY_OUTPUT = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project\output_clustered_vibgyor.ply"

# ==== CALIBRATION ====
T_cam_to_velo = np.array([
    [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
    [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
    [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
    [0, 0, 0, 1]
])
T_velo_to_cam0 = np.linalg.inv(T_cam_to_velo)

P = np.array([
    [552.554261, 0.0, 682.049453, 0.0],
    [0.0, 552.554261, 238.769549, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

# ==== FUNCTIONS ====
def load_lidar_bin(bin_path):
    lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return lidar[:, :3]

def project_to_image(pts_3d, P):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d_hom = pts_3d_hom @ P.T
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    return pts_2d

# ==== MAIN ====
image = cv2.imread(IMAGE_PATH)
lidar = load_lidar_bin(LIDAR_BIN_PATH)

# Project to camera
lidar_hom = np.hstack((lidar, np.ones((lidar.shape[0], 1))))
pts_cam = lidar_hom @ T_velo_to_cam0.T
valid = pts_cam[:, 2] > 0
pts_cam = pts_cam[valid]
pts_2d = project_to_image(pts_cam[:, :3], P)
lidar_filtered = lidar[valid]

# Start all points as grey
colors = np.full((lidar_filtered.shape[0], 3), fill_value=[40, 40, 40], dtype=np.float32) / 255.0

# Run YOLO
model = YOLO(YOLO_MODEL_PATH)
results = model(image)[0]

# VIBGYOR color palette (in RGB)
vibgyor_colors = [
    (148, 0, 211),    # Violet
    (75, 0, 130),     # Indigo
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (255, 127, 0),    # Orange
    (255, 0, 0),      # Red
]
vibgyor_colors = np.array(vibgyor_colors[:], dtype=np.float32) / 255.0

color_index = 0

if results.masks is not None:
    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for i, cls in enumerate(classes):
        if cls == 2:  # car
            if color_index >= len(vibgyor_colors):
                break

            mask = masks[i]
            mask_resized = cv2.resize((mask > 0.5).astype(np.uint8), (image.shape[1], image.shape[0]))

            vib_color = vibgyor_colors[color_index]
            color_index += 1

            for j, (x, y) in enumerate(pts_2d.astype(int)):
                if 0 <= x < mask_resized.shape[1] and 0 <= y < mask_resized.shape[0]:
                    if mask_resized[y, x] > 0:
                        colors[j] = vib_color

# Save and show point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(PLY_OUTPUT, pcd)
print("✅ Saved VIBGYOR-colored point cloud to:", PLY_OUTPUT)

# Show in Open3D
o3d.visualization.draw_geometries([pcd])
