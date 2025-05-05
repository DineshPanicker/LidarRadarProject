import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================= CONFIGURATION =============================

DATASET_PATH = r"A:\RWU\Second Sem\Lidar and Radar\KITTI-360_sample\Lidar-Project"

LIDAR_DIR = os.path.join(DATASET_PATH, "data_3d_raw", "2013_05_28_drive_0000_sync", "velodyne_points", "data")
IMAGE_DIR = os.path.join(DATASET_PATH, "data_2d_raw", "2013_05_28_drive_0000_sync", "image_00", "data_rect")
CALIB_DIR = os.path.join(DATASET_PATH, "calibration")

OUTPUT_DIR = os.path.join(DATASET_PATH, "output_yolo_car_projection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL_PATH = r"A:\RWU\Second Sem\Lidar and Radar\yolov8s-seg.pt"
CAR_CLASS_ID = 2  # COCO class id for 'car'

# ============================= FUNCTIONS =============================

def load_calibration(calib_dir):
    with open(os.path.join(calib_dir, "calib_cam_to_velo.txt")) as f:
        vals = list(map(float, f.readline().strip().split()))
    T_cam_to_velo = np.eye(4)
    T_cam_to_velo[:3, :4] = np.array(vals).reshape(3, 4)
    T_velo_to_cam0 = np.linalg.inv(T_cam_to_velo)

    with open(os.path.join(calib_dir, "perspective.txt")) as f:
        lines = f.readlines()

    P_rect_flat = None
    for line in lines:
        tokens = line.strip().split(":")
        if len(tokens) != 2:
            continue
        key, values = tokens
        values = list(map(float, values.strip().split()))
        if key == 'P_rect_00':
            P_rect_flat = values

    if P_rect_flat is None:
        raise ValueError("P_rect_00 not found!")

    P_rect_00 = np.array(P_rect_flat).reshape(3, 4)

    return T_velo_to_cam0, P_rect_00

def load_lidar_bin(bin_path):
    lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return lidar[:, :3]

def project_to_image(pts_3d, P):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d_hom = pts_3d_hom @ P.T
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    return pts_2d

def run_yolo_segmentation(image, model_path):
    model = YOLO(model_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, verbose=False)
    return results

def assign_points_to_masks(img_pts, masks, img_shape):
    assignment = -np.ones(len(img_pts), dtype=int)
    if not masks:
        return assignment

    valid_pts = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < img_shape[1]) & \
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < img_shape[0])

    img_pts_valid = img_pts[valid_pts].astype(int)

    for idx, mask in enumerate(masks):
        resized_mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
        x_coords = img_pts_valid[:, 0]
        y_coords = img_pts_valid[:, 1]
        inside = resized_mask[y_coords, x_coords] > 0.5
        assignment_idx = np.where(valid_pts)[0][inside]
        assignment[assignment_idx] = idx

    return assignment

def create_depth_map(img_shape, pts_2d, depths, selected_indices=None):
    depth_map = np.zeros(img_shape[:2], dtype=np.uint8)
    depths_selected = depths if selected_indices is None else depths[selected_indices]
    pts_selected = pts_2d if selected_indices is None else pts_2d[selected_indices]

    if len(depths_selected) == 0:
        return depth_map  # Return empty if no points

    depths_normalized = np.clip((depths_selected - depths_selected.min()) / (depths_selected.max() - depths_selected.min()) * 255, 0, 255).astype(np.uint8)

    for (x, y), d in zip(pts_selected, depths_normalized):
        x, y = int(x), int(y)
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            depth_map[y, x] = d

    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_colored

def draw_lidar_on_image(img, pts_2d, depths, selected_indices=None, color_map=cv2.COLORMAP_JET, alpha=0.8):
    img_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    depths_selected = depths if selected_indices is None else depths[selected_indices]
    pts_selected = pts_2d if selected_indices is None else pts_2d[selected_indices]

    if len(depths_selected) == 0:
        return cv2.cvtColor(img_with_alpha, cv2.COLOR_BGRA2BGR)

    depths_normalized = np.clip((depths_selected - depths_selected.min()) / (depths_selected.max() - depths_selected.min()) * 255, 0, 255).astype(np.uint8)

    for i, pt in enumerate(pts_selected):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color = cv2.applyColorMap(np.array([[depths_normalized[i]]], dtype=np.uint8), color_map)[0][0]
            img_with_alpha[y, x] = (*color[:3], int(alpha * 255))

    return cv2.cvtColor(img_with_alpha, cv2.COLOR_BGRA2BGR)

def save_combined(depth_img, overlay_img, save_path, sequence_id, frame_id):
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))

    axs[0].imshow(cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title("Projected Depth")

    axs[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axs[1].axis('off')
    axs[1].set_title("Projected Depth Overlaid on Image")

    fig.suptitle(f"Sequence {sequence_id}, Camera 00, Frame {frame_id}", fontsize=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()

# ============================= MAIN EXECUTION =============================

T_velo_to_cam0, P_rect_00 = load_calibration(CALIB_DIR)

lidar_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith('.bin')])

for lidar_file in lidar_files:
    scene_id = lidar_file.replace(".bin", "")

    lidar_path = os.path.join(LIDAR_DIR, lidar_file)
    img_path = os.path.join(IMAGE_DIR, f"{scene_id}.png")

    if not os.path.exists(img_path):
        print(f"⚠️ Warning: No image for {scene_id}. Skipping...")
        continue

    image = cv2.imread(img_path)
    lidar_points = load_lidar_bin(lidar_path)

    # Project lidar points
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    pts_cam0 = lidar_hom @ T_velo_to_cam0.T
    pts_cam0 = pts_cam0[pts_cam0[:, 2] > 0]
    pts_2d = project_to_image(pts_cam0[:, :3], P_rect_00)
    depths = pts_cam0[:, 2]

    # YOLO segmentation
    results = run_yolo_segmentation(image, YOLO_MODEL_PATH)

    car_masks = []
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        for idx, cls_id in enumerate(classes):
            if int(cls_id) == CAR_CLASS_ID:
                car_masks.append(masks[idx])

    # Assign lidar points to car masks
    assignment = assign_points_to_masks(pts_2d, car_masks, image.shape)
    car_indices = np.where(assignment >= 0)[0]

    if car_indices.size == 0:
        print(f"⚠️ Warning: No car points detected for {scene_id}. Skipping frame.")
        continue

    # Create images
    depth_img = create_depth_map(image.shape, pts_2d, depths, selected_indices=car_indices)
    overlay_img = draw_lidar_on_image(image, pts_2d, depths, selected_indices=car_indices)

    # Save and display
    save_path = os.path.join(OUTPUT_DIR, f"{scene_id}_combined.png")
    save_combined(depth_img, overlay_img, save_path, sequence_id="0000", frame_id=scene_id)

    print(f"✅ Saved combined for frame {scene_id}")

print("\n🎯 All frames processed successfully!")
