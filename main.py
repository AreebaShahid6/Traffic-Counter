import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time

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
if device == "cuda":
    model.half()   # FP16 on GPU — big speed boost

# =========================================================
# VIDEO
# =========================================================
video_path = r"C:\Users\Uzair Ahmad\Downloads\traffic-counter\traffic-videoo.mp4"

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
    2: (255, 100, 255),    # Bright Magenta for Car
    5: (100, 255, 100),    # Bright Green for Bus
    7: (255, 100, 100)     # Bright Red for Truck
}

# =========================================================
# COUNTERS
# =========================================================
counts_in  = defaultdict(int)
counts_out = defaultdict(int)

counted_in_ids  = set()
counted_out_ids = set()

track_history    = defaultdict(lambda: deque(maxlen=30))
vehicle_speeds   = defaultdict(list)
avg_speeds       = defaultdict(list)
total_vehicles_detected = set()

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
TOP_BAR_H     = 70
PANEL_H       = 310
CENTER_X      = 640

# =========================================================
# COUNTING LINE
# =========================================================
line_y   = 390
left_x1  = 0
left_x2  = CENTER_X - 10
right_x1 = CENTER_X + 10
right_x2 = FRAME_WIDTH

# =========================================================
# BOX SHRINK FACTOR
# =========================================================
BOX_SHRINK = 0.15

# =========================================================
# SPEED OPTIMIZATION — pre-build panel overlay ONCE
# =========================================================
panel_overlay = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
cv2.rectangle(panel_overlay, (0, 0),                    (FRAME_WIDTH, TOP_BAR_H), (20, 20, 40),  -1)
cv2.rectangle(panel_overlay, (0, TOP_BAR_H),             (LEFT_PANEL_W, PANEL_H),  (35, 15, 15),  -1)
cv2.rectangle(panel_overlay, (RIGHT_PANEL_X, TOP_BAR_H), (FRAME_WIDTH, PANEL_H),   (15, 35, 15),  -1)

# =========================================================
# MAIN LOOP
# =========================================================
frame_number   = 0
FRAME_SKIP     = 2
fps_start_time = time.time()
fps_counter    = 0
current_fps    = 0

while True:

    ret, frame = cap.read()
    if not ret:
        print("Video Finished")
        break

    frame_number += 1

    if frame_number % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    fps_counter += 1
    if fps_counter >= 30:
        current_fps    = fps_counter / (time.time() - fps_start_time)
        fps_start_time = time.time()
        fps_counter    = 0

    # =====================================================
    # TRACKING  — imgsz 480 instead of 640 (faster, same quality)
    # =====================================================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        imgsz=480,
        half=(device == "cuda"),
        verbose=False
    )

    r = results[0]

    # =====================================================
    # DRAW PANELS — use pre-built overlay (no rebuild each frame)
    # =====================================================
    frame = cv2.addWeighted(panel_overlay, 0.80, frame, 0.20, 0)

    # Panel borders with glow effect - ultra thin
    cv2.rectangle(frame, (0, 0),                    (FRAME_WIDTH, TOP_BAR_H), (100, 200, 255), 1)
    cv2.rectangle(frame, (0, TOP_BAR_H),             (LEFT_PANEL_W, PANEL_H),  (100, 100, 255), 1)
    cv2.rectangle(frame, (RIGHT_PANEL_X, TOP_BAR_H), (FRAME_WIDTH, PANEL_H),   (0, 255, 100),   1)

    # =====================================================
    # COUNTING LINES — exact same glow as original
    # =====================================================
    cv2.line(frame, (left_x1,  line_y), (left_x2,  line_y), (0, 0, 40),       8)
    cv2.line(frame, (left_x1,  line_y), (left_x2,  line_y), (0, 0, 80),       5)
    cv2.line(frame, (left_x1,  line_y), (left_x2,  line_y), (50, 100, 200),   3)
    cv2.line(frame, (left_x1,  line_y), (left_x2,  line_y), (100, 150, 255),  1)

    cv2.line(frame, (right_x1, line_y), (right_x2, line_y), (0, 40, 0),    8)
    cv2.line(frame, (right_x1, line_y), (right_x2, line_y), (0, 80, 0),    5)
    cv2.line(frame, (right_x1, line_y), (right_x2, line_y), (0, 150, 50),  3)
    cv2.line(frame, (right_x1, line_y), (right_x2, line_y), (0, 255, 100), 1)

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

            # bx/by prefix to avoid variable conflict with counting
            bx1, by1, bx2, by2 = map(int, box)

            w  = bx2 - bx1;  h  = by2 - by1
            dx = int(w * BOX_SHRINK);  dy = int(h * BOX_SHRINK)
            bx1 += dx;  bx2 -= dx
            by1 += dy;  by2 -= dy

            if (bx2 - bx1) < 20 or (by2 - by1) < 20:
                continue

            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)

            vehicle_name = class_names[cls]
            color        = class_colors[cls]

            total_vehicles_detected.add(track_id)

            track_history[track_id].append((cx, cy))
            points = list(track_history[track_id])

            if len(points) >= 5:
                dist  = np.sqrt((points[-1][0] - points[-5][0])**2 +
                                (points[-1][1] - points[-5][1])**2)
                speed = dist / 5
                vehicle_speeds[track_id].append(speed)
                avg_speeds[vehicle_name].append(speed)

            # Trail — glow + main (original logic)
            if len(points) > 1:
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    thickness = max(3, int(np.sqrt(float(i + 1)) * 0.8))
                    alpha = (i / len(points)) * 0.3
                    glow_color = tuple([int(c * alpha) for c in color])
                    cv2.line(frame, points[i - 1], points[i], glow_color, thickness)

                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    thickness = max(1, int(np.sqrt(float(i + 1)) * 0.5))
                    alpha = i / len(points)
                    trail_color = tuple([int(c * alpha) for c in color])
                    cv2.line(frame, points[i - 1], points[i], trail_color, thickness)

            # Neon box — original
            cv2.rectangle(frame, (bx1-3, by1-3), (bx2+3, by2+3), tuple([int(c*0.3) for c in color]), 1)
            cv2.rectangle(frame, (bx1-2, by1-2), (bx2+2, by2+2), tuple([int(c*0.5) for c in color]), 1)
            cv2.rectangle(frame, (bx1-1, by1-1), (bx2+1, by2+1), tuple([int(c*0.7) for c in color]), 1)
            cv2.rectangle(frame, (bx1+1, by1+1), (bx2+1, by2+1), (0, 0, 0), 1)
            cv2.rectangle(frame, (bx1,   by1),   (bx2,   by2),   color, 1)

            # Corner markers — original
            corner_len = 10
            cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), tuple([int(c*0.5) for c in color]), 4)
            cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), tuple([int(c*0.5) for c in color]), 4)

            cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), color, 2)
            cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), color, 2)
            cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), color, 2)
            cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), color, 2)
            cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), color, 2)
            cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), color, 2)
            cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), color, 2)
            cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), color, 2)

            # Pulsing center dot — original
            pulse_size = 4 + int(1 * np.sin(frame_number * 0.2))
            cv2.circle(frame, (cx, cy), pulse_size + 3, tuple([int(c*0.3) for c in color]), 1)
            cv2.circle(frame, (cx, cy), pulse_size + 2, tuple([int(c*0.5) for c in color]), 1)
            cv2.circle(frame, (cx, cy), pulse_size + 1, tuple([int(c*0.7) for c in color]), 1)
            cv2.circle(frame, (cx, cy), pulse_size,     (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), pulse_size - 1, color,     -1)

            # Label — original
            label = f"{vehicle_name} #{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_padding = 6
            cv2.rectangle(frame, (bx1-2, by1 - label_h - label_padding - 2),
                          (bx1 + label_w + label_padding + 2, by1+2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bx1, by1 - label_h - label_padding),
                          (bx1 + label_w + label_padding, by1), color, -1)
            cv2.putText(frame, label, (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # -----------------------------------------------
            # COUNTING — fixed (last 2 points, no var conflict)
            # -----------------------------------------------
            if len(points) >= 2:
                prev_cy = points[-2][1]
                curr_cy = points[-1][1]

                if prev_cy < line_y <= curr_cy:
                    if track_id not in counted_in_ids:
                        counted_in_ids.add(track_id)
                        counts_in[vehicle_name] += 1

                elif prev_cy > line_y >= curr_cy:
                    if track_id not in counted_out_ids:
                        counted_out_ids.add(track_id)
                        counts_out[vehicle_name] += 1

    # =====================================================
    # TOTALS
    # =====================================================
    total_in  = sum(counts_in.values())
    total_out = sum(counts_out.values())

    # =====================================================
    # TOP BAR — original
    # =====================================================
    cv2.putText(frame, "AI TRAFFIC VEHICLE COUNTER",
                (270, 38), cv2.FONT_HERSHEY_DUPLEX, 1.05, (100, 200, 255), 3)
    cv2.putText(frame, f"Frame: {frame_number} | FPS: {current_fps:.1f}",
                (450, 62), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 255, 255), 2)

    # =====================================================
    # LEFT PANEL — OUT COUNT
    # =====================================================
    cv2.putText(frame, "OUT COUNT",
                (28, TOP_BAR_H + 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 150, 255), 2)
    cv2.line(frame, (8, TOP_BAR_H + 55), (LEFT_PANEL_W - 8, TOP_BAR_H + 55), (150, 150, 255), 1)

    start_y = TOP_BAR_H + 100
    gap     = 45

    for i, cls_id in enumerate(vehicle_classes):
        name = class_names[cls_id]
        cv2.putText(frame, f"{name}:  {counts_out[name]}",
                    (12, start_y + i * gap),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, class_colors[cls_id], 2)

    cv2.putText(frame, f"TOTAL:  {total_out}",
                (12, PANEL_H - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

    # =====================================================
    # RIGHT PANEL — IN COUNT
    # =====================================================
    cv2.putText(frame, "IN COUNT",
                (1055, TOP_BAR_H + 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 150), 2)
    cv2.line(frame, (RIGHT_PANEL_X + 8, TOP_BAR_H + 55),
             (FRAME_WIDTH - 8, TOP_BAR_H + 55), (0, 255, 150), 1)

    for i, cls_id in enumerate(vehicle_classes):
        name = class_names[cls_id]
        cv2.putText(frame, f"{name}:  {counts_in[name]}",
                    (1050, start_y + i * gap),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, class_colors[cls_id], 2)

    cv2.putText(frame, f"TOTAL:  {total_in}",
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
