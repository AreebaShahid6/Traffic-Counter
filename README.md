# 🚦 AI Traffic Vehicle Counter using YOLO11, OpenCV & ByteTrack

An advanced AI-powered traffic monitoring and vehicle counting system developed using **YOLO11**, **OpenCV**, and **ByteTrack** for real-time vehicle detection, tracking, and directional traffic analytics.

This project is designed to analyze highway or road traffic footage and automatically detect, track, and count vehicles moving in different directions with a modern live analytics dashboard.

The system currently supports:

- 🚗 Cars
- 🚌 Buses
- 🚚 Trucks

with separate **IN** and **OUT** traffic counting.

---

# 📌 Project Overview

Traffic monitoring is one of the most important applications of Computer Vision in modern smart city infrastructure.

This project demonstrates how AI can be used for:

- Real-time traffic surveillance
- Vehicle movement analysis
- Direction-based vehicle counting
- Traffic flow analytics
- Intelligent transportation systems

The system uses the powerful **YOLO11 object detection model** combined with **ByteTrack multi-object tracking** to maintain unique IDs for each vehicle and accurately count vehicles crossing predefined counting lines.

---

# ✨ Features

## ✅ Real-Time Vehicle Detection

The project uses **YOLO11 Medium (`yolo11m.pt`)** for high-speed and accurate vehicle detection.

Supported vehicle classes:

- Car
- Bus
- Truck

---

## ✅ Multi-Object Tracking

Integrated with **ByteTrack** to:

- Assign unique IDs to vehicles
- Maintain tracking consistency
- Prevent duplicate counting

---

## ✅ Direction-Based Vehicle Counting

The system intelligently analyzes movement trajectories to determine:

- Vehicles entering the scene (**IN COUNT**)
- Vehicles leaving the scene (**OUT COUNT**)

Each vehicle is counted only once.

---

## ✅ Live Analytics Dashboard

Professional UI overlay includes:

- Vehicle counters
- IN / OUT analytics
- Total vehicle count
- Frame counter
- Detection labels
- Tracking IDs
- Counting lines

---

## ✅ Optimized Bounding Boxes

Bounding boxes are slightly reduced inward for:

- More accurate center-point tracking
- Improved counting precision
- Reduced false line crossing

---

## ✅ GPU Acceleration Support

The project automatically uses CUDA if available for faster inference.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| OpenCV | Video Processing & Visualization |
| YOLO11 | Object Detection |
| ByteTrack | Multi-Object Tracking |
| PyTorch | Deep Learning Backend |
| Ultralytics | YOLO Framework |

---

# 📂 Project Structure

```bash
AI-Traffic-Vehicle-Counter/
│
├── video/
│   └── traffic_video.mp4
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
git clone https://github.com/your-username/AI-Traffic-Vehicle-Counter.git
```

---

## 2️⃣ Move into Project Directory

```bash
cd AI-Traffic-Vehicle-Counter
```

---

## 3️⃣ Create Virtual Environment (Optional but Recommended)

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

Create a `requirements.txt` file:

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
|----------|---------------|
| 2 | Car |
| 5 | Bus |
| 7 | Truck |

---

# ⚙️ How the System Works

---

# 1️⃣ Video Input

The system reads a traffic video using OpenCV:

```python
cap = cv2.VideoCapture(video_path)
```

Supported inputs:

- Highway footage
- CCTV videos
- Drone traffic recordings
- Road surveillance videos

---

# 2️⃣ Object Detection using YOLO11

YOLO11 detects vehicles frame-by-frame.

```python
model = YOLO("yolo11m.pt")
```

Detection settings:

```python
conf=0.4
iou=0.5
imgsz=640
```

---

# 3️⃣ Multi-Object Tracking using ByteTrack

ByteTrack assigns a unique ID to each detected vehicle.

```python
tracker="bytetrack.yaml"
```

This ensures:

- Stable tracking
- No duplicate counting
- Smooth object association

---

# 4️⃣ Vehicle Center Point Calculation

The center point of each vehicle is calculated:

```python
cx = int((x1 + x2) / 2)
cy = int((y1 + y2) / 2)
```

These points are used for movement analysis.

---

# 5️⃣ Track History Storage

Vehicle movement history is stored:

```python
track_history[track_id].append((cx, cy))
```

This allows the system to determine direction.

---

# 6️⃣ Direction Detection

Direction is calculated using trajectory history:

```python
direction = points[-1][1] - points[0][1]
```

### Logic:

| Direction Value | Movement |
|----------------|----------|
| Positive | Moving Down |
| Negative | Moving Up |

---

# 7️⃣ Vehicle Counting Logic

Vehicles crossing predefined counting lines are counted.

### Left Side Logic

Vehicles moving downward across the line:

```python
if direction > 15 and cy > line_y:
```

counted as:

✅ IN Traffic

---

### Right Side Logic

Vehicles moving upward across the line:

```python
if direction < -15 and cy < line_y:
```

counted as:

✅ OUT Traffic

---

# 🖥️ User Interface Components

---

# 🔹 Top Information Bar

Displays:

- Project title
- Current frame number

---

# 🔹 Left Dashboard Panel

Shows:

## IN COUNT

- Car count
- Bus count
- Truck count
- Total incoming vehicles

---

# 🔹 Right Dashboard Panel

Shows:

## OUT COUNT

- Car count
- Bus count
- Truck count
- Total outgoing vehicles

---

# 🔹 Detection Overlay

Displays:

- Bounding boxes
- Vehicle labels
- Tracking IDs
- Vehicle center points

---

# 🎨 UI Design Features

## ✅ Semi-Transparent Dashboard

Created using:

```python
cv2.addWeighted()
```

Provides a modern visual appearance.

---

## ✅ Colored Vehicle Labels

| Vehicle | Color |
|---------|--------|
| Car | Blue |
| Bus | Red |
| Truck | Orange |

---

## ✅ Optimized Bounding Boxes

Bounding boxes are shrunk inward:

```python
BOX_SHRINK = 0.15
```

Benefits:

- Better center accuracy
- Reduced overlap errors
- Improved counting precision

---

# 📏 Counting Line Configuration

The counting line position:

```python
line_y = 370
```

The screen is divided into:

- Left Road
- Right Road

for directional traffic analysis.

---

# ⚡ Performance Optimization

The project includes several optimizations:

- GPU acceleration
- Reduced detection noise
- Bounding box filtering
- Frame resizing
- Stable object tracking

---

# 🧪 Tested Environment

| Component | Version |
|-----------|----------|
| Python | 3.10+ |
| OpenCV | Latest |
| PyTorch | Latest |
| CUDA | Supported |
| Windows | 10 / 11 |

---

# 📹 Example Use Cases

This project can be used for:

- Smart traffic systems
- Traffic flow monitoring
- Highway analytics
- Smart city infrastructure
- AI surveillance systems
- Research projects
- Computer vision learning

---

# 📸 Output Preview

Add screenshots or GIFs here.

Example:

```markdown
![Demo](screenshots/preview.png)
```

or

```markdown
![Demo GIF](screenshots/demo.gif)
```

---

# 🔥 Future Improvements

Planned future upgrades:

- 🚀 Speed estimation
- 🚀 Lane detection
- 🚀 Vehicle re-identification
- 🚀 Heatmap visualization
- 🚀 Traffic density analysis
- 🚀 Live CCTV stream integration
- 🚀 Web dashboard
- 🚀 Database storage
- 🚀 Cloud deployment
- 🚀 Multi-camera support

---

# 🤝 Contributing

Contributions are welcome!

If you would like to improve this project:

---

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

- Use
- Modify
- Distribute

with proper attribution.

---

# 👨‍💻 Author

## Your Name

AI & Computer Vision Developer

### GitHub

https://github.com/your-username

### LinkedIn

https://linkedin.com/in/your-profile

### Email

your-email@example.com

---

# ⭐ Support the Project

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Share with others
- 🚀 Contribute improvements

---

# 🙌 Acknowledgements

Special thanks to:

- Ultralytics YOLO
- OpenCV Community
- PyTorch Team
- ByteTrack Developers

for making advanced computer vision accessible.

---

# 📢 Final Note

This project demonstrates the power of combining:

- Deep Learning
- Object Detection
- Multi-Object Tracking
- Computer Vision

to build intelligent real-world AI traffic monitoring systems.

---
