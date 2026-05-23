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
# VEHICLE CLASSES
# =========================================================
vehicle_classes = [2, 3, 5, 7]

class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# =========================================================
# COLORS
# =========================================================
class_colors = {
    2: (255, 0, 0),
    3: (0, 255, 0),
    5: (0, 0, 255),
    7: (0, 165, 255)
}

# =========================================================
# COUNTERS
# =========================================================
counts_in = defaultdict(int)
counts_out = defaultdict(int)

counted_in_ids = set()
counted_out_ids = set()

track_history = defaultdict(list)

# =========================================================
# WINDOW
# =========================================================
cv2.namedWindow("Traffic Counter", cv2.WINDOW_NORMAL)

# =========================================================
# FRAME SIZE
# =========================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =========================================================
# PANELS
# =========================================================
LEFT_PANEL_W = 240
RIGHT_PANEL_X = 1040
PANEL_H = 285

CENTER_X = 640

# =========================================================
# COUNTING LINES
# =========================================================
left_line_y = 500
right_line_y = 360  # moved higher

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
    # PANELS
    # =====================================================
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (LEFT_PANEL_W, PANEL_H),
        (10, 10, 10),
        -1
    )

    cv2.rectangle(
        overlay,
        (RIGHT_PANEL_X, 0),
        (FRAME_WIDTH, PANEL_H),
        (10, 10, 10),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.85,
        frame,
        0.15,
        0
    )

    # =====================================================
    # PANEL BORDERS
    # =====================================================
    cv2.rectangle(
        frame,
        (0, 0),
        (LEFT_PANEL_W, PANEL_H),
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        frame,
        (RIGHT_PANEL_X, 0),
        (FRAME_WIDTH, PANEL_H),
        (0, 0, 255),
        2
    )

    # =====================================================
    # LEFT GREEN LINE
    # =====================================================
    left_x1 = 0
    left_x2 = 630

    cv2.line(
        frame,
        (left_x1, left_line_y),
        (left_x2, left_line_y),
        (0, 255, 0),
        6
    )

    # =====================================================
    # RIGHT RED LINE
    # =====================================================
    right_x1 = 665
    right_x2 = 1270

    cv2.line(
        frame,
        (right_x1, right_line_y),
        (right_x2, right_line_y),
        (0, 0, 255),
        6
    )

    # =====================================================
    # DETECTIONS
    # =====================================================
    if r.boxes is not None and r.boxes.id is not None:

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            cls = int(cls)
            track_id = int(track_id)

            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            vehicle_name = class_names[cls]
            color = class_colors[cls]

            # =================================================
            # TRACK HISTORY
            # =================================================
            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) > 20:
                track_history[track_id].pop(0)

            points = track_history[track_id]

            # =================================================
            # DRAW BOX
            # =================================================
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # CENTER DOT
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # LABEL
            cv2.putText(
                frame,
                f"{vehicle_name} {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                color,
                2
            )

            # =================================================
            # COUNTING
            # =================================================
            if len(points) >= 6:

                direction = points[-1][1] - points[0][1]

                # LEFT ROAD
                if cx < CENTER_X:

                    if direction > 15 and cy > left_line_y:

                        if track_id not in counted_in_ids:

                            counted_in_ids.add(track_id)
                            counts_in[vehicle_name] += 1

                # RIGHT ROAD
                else:

                    if direction < -15 and cy < right_line_y:

                        if track_id not in counted_out_ids:

                            counted_out_ids.add(track_id)
                            counts_out[vehicle_name] += 1

    # =====================================================
    # TOTALS
    # =====================================================
    total_in = sum(counts_in.values())
    total_out = sum(counts_out.values())

    # =====================================================
    # TITLE
    # =====================================================
    cv2.putText(
        frame,
        "AI TRAFFIC VEHICLE COUNTER",
        (260, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (510, 85),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        (0, 220, 255),
        2
    )

    # =====================================================
    # LEFT PANEL
    # =====================================================
    cv2.putText(
        frame,
        "IN COUNT",
        (28, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        (8, 58),
        (LEFT_PANEL_W - 8, 58),
        (0, 180, 0),
        2
    )

    start_y = 100
    gap = 45

    for i, cls_id in enumerate(vehicle_classes):

        name = class_names[cls_id]

        cv2.putText(
            frame,
            f"{name}:  {counts_in[name]}",
            (12, start_y + (i * gap)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            class_colors[cls_id],
            2
        )

    cv2.putText(
        frame,
        f"TOTAL:  {total_in}",
        (12, 275),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    # =====================================================
    # RIGHT PANEL
    # =====================================================
    cv2.putText(
        frame,
        "OUT COUNT",
        (1055, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.line(
        frame,
        (RIGHT_PANEL_X + 8, 58),
        (FRAME_WIDTH - 8, 58),
        (180, 0, 0),
        2
    )

    for i, cls_id in enumerate(vehicle_classes):

        name = class_names[cls_id]

        cv2.putText(
            frame,
            f"{name}:  {counts_out[name]}",
            (1050, start_y + (i * gap)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            class_colors[cls_id],
            2
        )

    cv2.putText(
        frame,
        f"TOTAL:  {total_out}",
        (1050, 275),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    # =====================================================
    # SHOW
    # =====================================================
    cv2.imshow("Traffic Counter", frame)

    # =====================================================
    # EXIT
    # =====================================================
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC Pressed")
        break

# =========================================================
# RELEASE
# =========================================================
cap.release()
cv2.destroyAllWindows()