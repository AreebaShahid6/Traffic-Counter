import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict

# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO("yolo11m.pt")
model.to(device)

# =========================================================
# VIDEO
# =========================================================
video_path = "video/YTDown_YouTube_4K-Video-of-Highway-Traffic_Media_KBsqQez-O4w_001_1080p.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

# =========================================================
# VEHICLE CLASSES  — Motorcycle COMPLETELY REMOVED
# =========================================================
vehicle_classes = [2, 5, 7]   # Car, Bus, Truck ONLY

class_names = {
    2: "Car",
    5: "Bus",
    7: "Truck"
}

class_colors = {
    2: (255, 0,   0),
    5: (0,   0, 255),
    7: (0, 165, 255)
}

# =========================================================
# COUNTERS
# =========================================================
counts_in  = defaultdict(int)
counts_out = defaultdict(int)

counted_in_ids  = set()
counted_out_ids = set()

track_history = defaultdict(list)

# =========================================================
# WINDOW
# =========================================================
cv2.namedWindow("Traffic Counter", cv2.WINDOW_NORMAL)

# =========================================================
# FRAME SIZE
# =========================================================
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# =========================================================
# PANELS
# =========================================================
LEFT_PANEL_W  = 240
RIGHT_PANEL_X = 1040

TOP_BAR_H = 70
PANEL_H   = 310          # shorter panel — only 3 vehicle types now

CENTER_X  = 640

# =========================================================
# COUNTING LINES — moved up
# =========================================================
line_y   = 370
left_x1  = 0
left_x2  = CENTER_X - 10
right_x1 = CENTER_X + 10
right_x2 = FRAME_WIDTH

# =========================================================
# BOX SHRINK FACTOR  (0.0 = no shrink, 0.15 = 15% smaller)
# =========================================================
BOX_SHRINK = 0.15

# =========================================================
# MAIN LOOP
# =========================================================
frame_number = 0

while True:

    ret, frame = cap.read()
    if not ret:
        print("Video Finished")
        break

    frame_number += 1
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # =====================================================
    # TRACKING
    # =====================================================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        imgsz=640,
        verbose=False
    )

    r = results[0]

    # =====================================================
    # DRAW PANELS + TOP BAR
    # =====================================================
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0),            (FRAME_WIDTH, TOP_BAR_H), (10, 10, 10), -1)
    cv2.rectangle(overlay, (0, TOP_BAR_H),    (LEFT_PANEL_W, PANEL_H),  (10, 10, 10), -1)
    cv2.rectangle(overlay, (RIGHT_PANEL_X, TOP_BAR_H), (FRAME_WIDTH, PANEL_H), (10, 10, 10), -1)

    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Panel borders
    cv2.rectangle(frame, (0, 0),            (FRAME_WIDTH, TOP_BAR_H), (80, 80, 80), 2)
    cv2.rectangle(frame, (0, TOP_BAR_H),    (LEFT_PANEL_W, PANEL_H),  (0, 255, 0), 2)
    cv2.rectangle(frame, (RIGHT_PANEL_X, TOP_BAR_H), (FRAME_WIDTH, PANEL_H), (0, 0, 255), 2)

    # =====================================================
    # COUNTING LINES
    # =====================================================
    cv2.line(frame, (left_x1,  line_y), (left_x2,  line_y), (0, 255, 0), 6)
    cv2.line(frame, (right_x1, line_y), (right_x2, line_y), (0, 0, 255), 6)

    # =====================================================
    # DETECTIONS
    # =====================================================
    if r.boxes is not None and r.boxes.id is not None:

        boxes   = r.boxes.xyxy.cpu().numpy()
        ids     = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            cls      = int(cls)
            track_id = int(track_id)

            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            # ---- SHRINK BOX inward from center ----
            w  = x2 - x1
            h  = y2 - y1
            dx = int(w * BOX_SHRINK)
            dy = int(h * BOX_SHRINK)
            x1 += dx;  x2 -= dx
            y1 += dy;  y2 -= dy

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            vehicle_name = class_names[cls]
            color        = class_colors[cls]

            # Track history
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 20:
                track_history[track_id].pop(0)
            points = track_history[track_id]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{vehicle_name} {track_id}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

            # Counting
            if len(points) >= 6:
                direction = points[-1][1] - points[0][1]

                if cx < CENTER_X:
                    if direction > 15 and cy > line_y:
                        if track_id not in counted_in_ids:
                            counted_in_ids.add(track_id)
                            counts_in[vehicle_name] += 1
                else:
                    if direction < -15 and cy < line_y:
                        if track_id not in counted_out_ids:
                            counted_out_ids.add(track_id)
                            counts_out[vehicle_name] += 1

    # =====================================================
    # TOTALS
    # =====================================================
    total_in  = sum(counts_in.values())
    total_out = sum(counts_out.values())

    # =====================================================
    # TOP BAR
    # =====================================================
    cv2.putText(frame, "AI TRAFFIC VEHICLE COUNTER",
                (270, 38), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_number}",
                (510, 62), cv2.FONT_HERSHEY_DUPLEX, 0.80, (0, 220, 255), 2)

    # =====================================================
    # LEFT PANEL  — Car / Bus / Truck only
    # =====================================================
    cv2.putText(frame, "IN COUNT",
                (28, TOP_BAR_H + 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
    cv2.line(frame, (8, TOP_BAR_H + 55), (LEFT_PANEL_W - 8, TOP_BAR_H + 55), (0, 180, 0), 2)

    start_y = TOP_BAR_H + 100
    gap     = 45

    for i, cls_id in enumerate(vehicle_classes):
        name = class_names[cls_id]
        cv2.putText(frame, f"{name}:  {counts_in[name]}",
                    (12, start_y + (i * gap)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, class_colors[cls_id], 2)

    cv2.putText(frame, f"TOTAL:  {total_in}",
                (12, PANEL_H - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

    # =====================================================
    # RIGHT PANEL  — Car / Bus / Truck only
    # =====================================================
    cv2.putText(frame, "OUT COUNT",
                (1050, TOP_BAR_H + 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    cv2.line(frame, (RIGHT_PANEL_X + 8, TOP_BAR_H + 55),
             (FRAME_WIDTH - 8, TOP_BAR_H + 55), (180, 0, 0), 2)

    for i, cls_id in enumerate(vehicle_classes):
        name = class_names[cls_id]
        cv2.putText(frame, f"{name}:  {counts_out[name]}",
                    (1050, start_y + (i * gap)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, class_colors[cls_id], 2)

    cv2.putText(frame, f"TOTAL:  {total_out}",
                (1050, PANEL_H - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

    # =====================================================
    # SHOW
    # =====================================================
    cv2.imshow("Traffic Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC Pressed")
        break

cap.release()
cv2.destroyAllWindows()