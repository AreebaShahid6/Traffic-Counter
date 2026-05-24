# 🚦 AI Traffic Vehicle Counter using YOLO11, OpenCV & ByteTrack

An advanced AI-powered traffic monitoring and vehicle counting system developed using **YOLO11**, **OpenCV**, and **ByteTrack** for real-time vehicle detection, tracking, speed estimation, and directional traffic analytics.

This project analyzes traffic footage and automatically detects, tracks, and counts vehicles moving in different directions with a modern neon-style analytics dashboard.

The system currently supports:

* 🚗 Cars
* 🚌 Buses
* 🚚 Trucks

with separate **IN** and **OUT** traffic counting.

---

# 📌 Project Overview

Traffic monitoring is one of the most important applications of Computer Vision in modern smart city infrastructure.

This project demonstrates how Artificial Intelligence can be used for:

* Real-time traffic surveillance
* Vehicle movement analysis
* Direction-based vehicle counting
* Traffic flow analytics
* Intelligent transportation systems
* Vehicle trajectory tracking
* Basic speed estimation

The system combines the power of **YOLO11 object detection** with **ByteTrack multi-object tracking** to maintain unique IDs for each vehicle and accurately count vehicles crossing predefined counting lines.

---

# ✨ Features

## ✅ Real-Time Vehicle Detection

Uses **YOLO11 Medium (`yolo11m.pt`)** for high-speed and accurate vehicle detection.

Supported vehicle classes:

* Car
* Bus
* Truck

---

## ✅ Multi-Object Tracking

Integrated with **ByteTrack** to:

* Assign unique IDs to vehicles
* Maintain stable tracking
* Prevent duplicate counting
* Track vehicle movement paths

---

## ✅ Direction-Based Vehicle Counting

The system intelligently analyzes movement trajectories to determine:

* Vehicles entering the scene (**IN COUNT**)
* Vehicles leaving the scene (**OUT COUNT**)

Each vehicle is counted only once using tracking IDs.

---

## ✅ Live Analytics Dashboard

Professional real-time UI overlay includes:

* IN / OUT counters
* Vehicle totals
* FPS monitor
* Frame counter
* Detection labels
* Tracking IDs
* Neon counting lines
* Vehicle trails

---

## ✅ Vehicle Trail Visualization

Each vehicle leaves a smooth motion trail showing movement history.

This helps visualize:

* Vehicle direction
* Motion flow
* Tracking stability

---

## ✅ Speed Estimation

The system estimates vehicle motion speed using movement distance between tracked points.

```python
dist = np.sqrt((points[-1][0] - points[-5][0])**2 +
               (points[-1][1] - points[-5][1])**2)
```

---

## ✅ GPU Acceleration Support

The project automatically detects CUDA availability and enables GPU acceleration.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

FP16 inference is automatically enabled on GPU for higher FPS performance.

---

# 🧠 Technologies Used

| Technology  | Purpose                          |
| ----------- | -------------------------------- |
| Python      | Core Programming                 |
| OpenCV      | Video Processing & Visualization |
| YOLO11      | Object Detection                 |
| ByteTrack   | Multi-Object Tracking            |
| PyTorch     | Deep Learning Backend            |
| Ultralytics | YOLO Framework                   |
| NumPy       | Mathematical Operations          |

---

# 📂 Project Structure

```bash
Traffic-Counter/
│
├── video/
│   └── traffic-video.mp4
│
├── screenshots/
│   └── preview.png
│
├── output/
│   └── result_video.mp4
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/AreebaShahid6/Traffic-Counter.git
```

---

## 2️⃣ Move into Project Directory

```bash
cd Traffic-Counter
```

---

## 3️⃣ Create Virtual Environment (Optional)

### Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

### Linux / macOS

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```txt
ultralytics
opencv-python
torch
torchvision
numpy
```

---

# ▶️ Running the Project

Run the Python script:

```bash
python main.py
```

---

# 🎯 Supported Vehicle Classes

The system uses COCO dataset class IDs.

| Class ID | Vehicle Type |
| -------- | ------------ |
| 2        | Car          |
| 5        | Bus          |
| 7        | Truck        |

---

# ⚙️ How the System Works

---

# 1️⃣ Video Input

Traffic footage is loaded using OpenCV:

```python
cap = cv2.VideoCapture(video_path)
```

Supported sources:

* Highway footage
* CCTV videos
* Drone footage
* Traffic surveillance cameras

---

# 2️⃣ Object Detection using YOLO11

YOLO11 detects vehicles frame-by-frame.

```python
model = YOLO("yolo11m.pt")
```

Detection configuration:

```python
conf=0.4
iou=0.5
imgsz=480
```

---

# 3️⃣ Multi-Object Tracking using ByteTrack

ByteTrack assigns a unique ID to every detected vehicle.

```python
tracker="bytetrack.yaml"
```

This ensures:

* Stable tracking
* Reduced ID switching
* Accurate counting

---

# 4️⃣ Vehicle Center Point Calculation

Vehicle center coordinates are calculated using:

```python
cx = int((x1 + x2) / 2)
cy = int((y1 + y2) / 2)
```

These points are used for movement analysis and counting.

---

# 5️⃣ Vehicle Trail History

The system stores previous vehicle positions:

```python
track_history[track_id].append((cx, cy))
```

This creates smooth visual movement trails.

---

# 6️⃣ Direction Detection Logic

Direction is determined using previous and current center positions.

```python
prev_cy = points[-2][1]
curr_cy = points[-1][1]
```

### Counting Logic

| Movement    | Action    |
| ----------- | --------- |
| Moving Down | IN Count  |
| Moving Up   | OUT Count |

---

# 7️⃣ Counting Line System

Vehicles crossing the counting line are counted automatically.

```python
line_y = 390
```

Each vehicle is counted only once using unique tracking IDs.

---

# 🖥️ User Interface Components

---

# 🔹 Top Information Bar

Displays:

* Project title
* Current frame number
* Live FPS

---

# 🔹 Left Dashboard Panel

Displays:

## OUT COUNT

* Car count
* Bus count
* Truck count
* Total outgoing vehicles

---

# 🔹 Right Dashboard Panel

Displays:

## IN COUNT

* Car count
* Bus count
* Truck count
* Total incoming vehicles

---

# 🔹 Detection Overlay

Displays:

* Neon bounding boxes
* Vehicle labels
* Tracking IDs
* Vehicle center points
* Motion trails

---

# 🎨 UI Design Features

## ✅ Neon Styled Interface

Modern glowing UI created using layered OpenCV drawing functions.

---

## ✅ Semi-Transparent Dashboard

Created using:

```python
cv2.addWeighted()
```

Provides a modern AI dashboard appearance.

---

## ✅ Colored Vehicle Labels

| Vehicle | Color   |
| ------- | ------- |
| Car     | Magenta |
| Bus     | Green   |
| Truck   | Red     |

---

## ✅ Optimized Bounding Boxes

Bounding boxes are reduced inward:

```python
BOX_SHRINK = 0.15
```

Benefits:

* Better center accuracy
* Reduced overlap errors
* Improved counting precision

---

# ⚡ Performance Optimization

The project includes multiple optimizations:

* CUDA GPU acceleration
* FP16 inference
* Frame skipping
* Reduced image size (`imgsz=480`)
* Pre-built overlay rendering
* Stable tracking system

---

# 🧪 Tested Environment

| Component | Version   |
| --------- | --------- |
| Python    | 3.10+     |
| OpenCV    | Latest    |
| PyTorch   | Latest    |
| CUDA      | Supported |
| Windows   | 10 / 11   |

---

# 📹 Example Use Cases

This project can be used for:

* Smart traffic systems
* Highway analytics
* AI surveillance systems
* Smart city infrastructure
* Traffic flow monitoring
* Computer vision research
* Vehicle movement analysis

---

# 📸 Output Preview

Add screenshots or GIFs here.

Example:

```markdown
![Demo](screenshots/preview.png)
```

---

# 🔥 Future Improvements

Planned future upgrades:

* 🚀 Real-world speed calculation (km/h)
* 🚀 Lane detection
* 🚀 Heatmap visualization
* 🚀 Traffic density estimation
* 🚀 Web dashboard
* 🚀 Database integration
* 🚀 Multi-camera support
* 🚀 Live CCTV stream support
* 🚀 Cloud deployment

---

# 🤝 Contributing

Contributions are welcome!

## Steps to Contribute

### 1️⃣ Fork the Repository

Click the **Fork** button on GitHub.

---

### 2️⃣ Create a New Branch

```bash
git checkout -b feature-name
```

---

### 3️⃣ Commit Changes

```bash
git commit -m "Added new feature"
```

---

### 4️⃣ Push Changes

```bash
git push origin feature-name
```

---

### 5️⃣ Open Pull Request

Submit your PR for review.

---

# 📜 License

This project is licensed under the MIT License.

You are free to:

* Use
* Modify
* Distribute

with proper attribution.

---

# 👨‍💻 Author

## Areeba Shahid

AI & Computer Vision Developer

### GitHub

[AreebaShahid6](https://github.com/AreebaShahid6?utm_source=chatgpt.com)

### Email

[shahidareeba922@gmail.com](mailto:shahidareeba922@gmail.com)

---

# ⭐ Support the Project

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🧠 Share with others
* 🚀 Contribute improvements

---

# 🙌 Acknowledgements

Special thanks to:

* [Ultralytics YOLO](https://www.ultralytics.com/?utm_source=chatgpt.com)
* [OpenCV](https://opencv.org/?utm_source=chatgpt.com)
* [PyTorch](https://pytorch.org/?utm_source=chatgpt.com)
* [ByteTrack](https://github.com/ifzhang/ByteTrack?utm_source=chatgpt.com)

for making advanced computer vision accessible.

---

# 📢 Final Note

This project demonstrates the power of combining:

* Deep Learning
* Object Detection
* Multi-Object Tracking
* Computer Vision
* Real-Time Analytics

to build intelligent AI-based traffic monitoring systems.
