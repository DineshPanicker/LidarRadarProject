import os
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# ========== CONFIGURATION ==========
DATASET_PATH = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project"

LIDAR_DIR = os.path.join(DATASET_PATH, "data_3d_raw", "2013_05_28_drive_0000_sync", "velodyne_points", "data")
IMAGE_DIR = os.path.join(DATASET_PATH, "data_2d_raw", "2013_05_28_drive_0000_sync", "image_00", "data_rect")
CALIB_DIR = os.path.join(DATASET_PATH, "calibration")

OUTPUT_IMAGE_DIR = os.path.join(DATASET_PATH, "output_yolo_image")
OUTPUT_PLY_DIR = os.path.join(DATASET_PATH, "output_yolo_ply")
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLY_DIR, exist_ok=True)

YOLO_MODEL_PATH = r"A:\RWU\Second Sem\Lidar and Radar\yolov8s-seg.pt"
CAR_CLASS_ID = 2  # COCO class ID for 'car'

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
    raise ValueError("P_rect_00 not found in perspective.txt")

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

def draw_on_image(img, pts_2d, assignment, num_masks):
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2BGRA)
    vibgyor = [(255,0,0),(255,165,0),(255,255,0),(0,255,0),(0,0,255),(75,0,130),(238,130,238)]
    for i, pt in enumerate(pts_2d.astype(int)):
        x, y = pt
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if assignment[i] >= 0:
                color = vibgyor[assignment[i] % len(vibgyor)]
            else:
                color = (50,50,50)
            vis[y, x] = (*color, 255)
    return cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)

def create_pcd(points, assignment, num_masks):
    vibgyor = np.array([
        [1, 0, 0], [1, 0.6, 0], [1, 1, 0],
        [0, 1, 0], [0, 0, 1], [0.29, 0, 0.51], [0.93, 0.5, 0.93]
    ])
    colors = np.ones((len(points), 3)) * 0.2  # default grey
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
    if not fname.endswith(".bin"): continue

    scene_id = fname.replace(".bin", "")
    lidar = load_lidar(os.path.join(LIDAR_DIR, fname))
    image = cv2.imread(os.path.join(IMAGE_DIR, f"{scene_id}.png"))
    if image is None:
        print(f"Skipping {scene_id} — no image.")
        continue

    # Project
    lidar_hom = np.hstack((lidar, np.ones((lidar.shape[0], 1))))
    pts_cam = lidar_hom @ T_velo_to_cam.T
    valid = pts_cam[:, 2] > 0
    pts_cam, lidar = pts_cam[valid], lidar[valid]
    pts_2d = project_points(pts_cam[:, :3], P_rect_00)

    # YOLOv8 segmentation
    results = model(image)[0]
    car_masks = []
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        car_masks = [masks[i] for i, c in enumerate(classes) if c == CAR_CLASS_ID]

    assignment = assign_points(pts_2d, car_masks, image.shape)
    num_cars = len(car_masks)

    # Save 2D image
    overlay_img = draw_on_image(image, pts_2d, assignment, num_cars)
    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, f"{scene_id}_yolo_overlay.png"), overlay_img)

    # Save 3D point cloud
    pcd = create_pcd(lidar, assignment, num_cars)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_PLY_DIR, f"{scene_id}_cars_colored.ply"), pcd)

    print(f"✅ Processed {scene_id}")

print("\n🎯 All frames processed successfully!")
