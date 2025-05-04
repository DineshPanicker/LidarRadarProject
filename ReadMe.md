# KITTI-360 Project: LiDAR and Camera Fusion with YOLO Segmentation

This project demonstrates LiDAR point cloud projection onto 2D camera images using the KITTI-360 dataset. It integrates instance segmentation (YOLOv8) to identify cars in the image, then assigns unique colors to 3D LiDAR points corresponding to each detected car. The final 3D point cloud is visualized in Open3D with vibrant VIBGYOR coloring for cars and grey/black for background points.

---

## 📁 Dataset Description

This sample of the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) dataset includes:

* 20 frames of images from Cam 1 and Cam 2
* Velodyne 3D point clouds
* Calibration files
* Custom 3D bounding boxes (`bboxes_3D_cam0`) provided in JSON format (indexed per frame)

Refer to the official [documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php) for full dataset structure.

---

## 🧠 Project Workflow

### 1. **Data Loading**

* Read LiDAR binary files (`.bin`) into Nx4 arrays (x, y, z, intensity)
* Read corresponding RGB images

### 2. **Calibration Parsing**

* Use `calib_cam_to_velo.txt` to transform LiDAR to camera coordinates
* Use `perspective.txt` for the projection matrix `P_rect_00`

### 3. **YOLOv8 Instance Segmentation**

* Run `yolov8s-seg.pt` model to segment cars from the image
* Assign each car a unique mask

### 4. **Projection & Filtering**

* Project LiDAR points to 2D using the projection matrix
* Filter points that fall inside each car's segmentation mask

### 5. **Color Coding**

* Assign VIBGYOR colors to each car cluster
* Color all non-car/background points as black or grey

### 6. **Visualization**

* Display projected points over 2D image (OpenCV)
* Export 3D colored point cloud as `.ply` and visualize using Open3D

---

## 🖼️ Output Samples

* `output_image_with_car_mask.png`: Overlaid segmentation masks and LiDAR points on RGB image
* `scene_id_3d.ply`: Colored 3D point cloud for each frame

---

## 🔧 Dependencies

```bash
pip install ultralytics open3d numpy opencv-python
```

Model file:

```
YOLO_MODEL_PATH = r"A:\RWU\Second Sem\Lidar and Radar\yolov8s-seg.pt"
```

---

## 📦 File Structure

```
Lidar-Project/
├── calibration/
│   ├── calib_cam_to_velo.txt
│   ├── perspective.txt
├── data_2d_raw/
│   └── ... image data ...
├── data_3d_raw/
│   └── ... velodyne bin files ...
├── bboxes_3D_cam0/
│   └── BBoxes_xxx.json
├── output_result/
│   ├── *.png
│   └── *.ply
└── main.py
```

---

## 🧩 Acknowledgments

* [KITTI-360 dataset](https://www.cvlibs.net/datasets/kitti-360/)
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Open3D](http://www.open3d.org/)

---

## ✍️ License

MIT License. For dataset use, refer to KITTI-360's CC-BY-NC-SA 3.0 License.

---

## 🙋‍♂️ Author

This project is implemented by a student at RWU as part of the "Lidar and Radar Systems" course. Reach out for questions or collaboration!
