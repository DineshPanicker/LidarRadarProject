import os
import numpy as np
import cv2
import json
import open3d as o3d
from ultralytics import YOLO

# ========== CONFIG ==========
DATASET_PATH = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project"
LIDAR_DIR = os.path.join(DATASET_PATH, "data_3d_raw", "2013_05_28_drive_0000_sync", "velodyne_points", "data")
IMAGE_DIR = os.path.join(DATASET_PATH, "data_2d_raw", "2013_05_28_drive_0000_sync", "image_00", "data_rect")
CALIB_DIR = os.path.join(DATASET_PATH, "calibration")
BBOX_DIR = os.path.join(DATASET_PATH, "bboxes_3D_cam0")
OUTPUT_PLY_DIR = os.path.join(DATASET_PATH, "output_yolo_gt_ply")
os.makedirs(OUTPUT_PLY_DIR, exist_ok=True)

YOLO_MODEL_PATH = os.path.join(DATASET_PATH, "yolov8s-seg.pt")
CAR_CLASS_ID = 2  # COCO ID for car

# ========== FUNCTIONS ==========
def load_calibration(calib_dir):
    with open(os.path.join(calib_dir, "calib_cam_to_velo.txt")) as f:
        vals = list(map(float, f.readline().strip().split()))
    T = np.eye(4)
    T[:3, :4] = np.array(vals).reshape(3, 4)
    T_velo_to_cam = np.linalg.inv(T)
    with open(os.path.join(calib_dir, "perspective.txt")) as f:
        for line in f:
            if "P_rect_00:" in line:
                P_vals = list(map(float, line.strip().split(":")[1].split()))
                P = np.array(P_vals).reshape(3, 4)
                return T_velo_to_cam, P
    raise ValueError("P_rect_00 not found")

def load_lidar(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def project_points(pts_3d, P):
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d_hom = pts_hom @ P.T
    return pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]

def assign_points(img_pts, masks, shape):
    assignment = -np.ones(len(img_pts), dtype=int)
    valid = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < shape[1]) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < shape[0])
    valid_pts = img_pts[valid].astype(int)
    for i, mask in enumerate(masks):
        resized = cv2.resize(mask, (shape[1], shape[0]))
        inside = resized[valid_pts[:, 1], valid_pts[:, 0]] > 0.5
        idxs = np.where(valid)[0][inside]
        assignment[idxs] = i
    return assignment

def load_ground_truth_boxes(bbox_dir, scene_id):
    json_path = os.path.join(bbox_dir, f"BBoxes_{int(scene_id)}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("boxes", [])

def create_pcd(points, assignment, num_masks):
    vibgyor = np.array([
        [1, 0, 0], [1, 0.6, 0], [1, 1, 0],
        [0, 1, 0], [0, 0, 1], [0.29, 0, 0.51], [0.93, 0.5, 0.93]
    ])
    colors = np.ones((len(points), 3)) * 0.2
    for i in range(num_masks):
        colors[assignment == i] = vibgyor[i % len(vibgyor)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# ========== MAIN ==========
T_velo_to_cam, P_rect_00 = load_calibration(CALIB_DIR)
model = YOLO(YOLO_MODEL_PATH)

for fname in sorted(os.listdir(LIDAR_DIR)):
    if not fname.endswith(".bin"):
        continue

    scene_id = fname.replace(".bin", "")
    lidar = load_lidar(os.path.join(LIDAR_DIR, fname))
    image = cv2.imread(os.path.join(IMAGE_DIR, f"{scene_id}.png"))
    if image is None:
        continue

    lidar_hom = np.hstack((lidar, np.ones((lidar.shape[0], 1))))
    pts_cam = lidar_hom @ T_velo_to_cam.T
    valid = pts_cam[:, 2] > 0
    pts_cam, lidar = pts_cam[valid], lidar[valid]
    pts_2d = project_points(pts_cam[:, :3], P_rect_00)

    results = model(image)[0]
    car_masks = []
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        car_masks = [masks[i] for i, c in enumerate(classes) if c == CAR_CLASS_ID]

    assignment = assign_points(pts_2d, car_masks, image.shape)
    np.save(os.path.join(OUTPUT_PLY_DIR, f"{scene_id}_assignment.npy"), assignment)
    num_cars = len(car_masks)

    pcd = create_pcd(lidar, assignment, num_cars)

    # Save .ply
    ply_path = os.path.join(OUTPUT_PLY_DIR, f"{scene_id}_yolo_gt.ply")
    o3d.io.write_point_cloud(ply_path, pcd)

    # Save GT bounding boxes (transformed to LiDAR space) to JSON
    gt_boxes_lidar = []
    boxes = load_ground_truth_boxes(BBOX_DIR, scene_id)
    for box in boxes:
        corners_cam0 = np.array(box["corners_cam0"])
        corners_hom = np.hstack((corners_cam0, np.ones((8, 1))))
        corners_lidar = (np.linalg.inv(T_velo_to_cam) @ corners_hom.T).T[:, :3]
        gt_boxes_lidar.append(corners_lidar.tolist()) #lodaing the GT boxes onto the LiDAR pointcloud

    gt_path = os.path.join(OUTPUT_PLY_DIR, f"{scene_id}_gt_boxes.json")
    with open(gt_path, "w") as f:
        json.dump(gt_boxes_lidar, f, indent=2)

    print(f"✅ Saved {scene_id} → PLY and GT boxes")

print("\n🎯 All frames done.")
