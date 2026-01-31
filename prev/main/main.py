from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from collections import defaultdict
import time
import datetime
from datetime import datetime
import os

model = YOLO("models/best_yolov8_improved.pt", task="detect")
class_names = model.names

colors = {
    "Creeping": {"normal": (0, 255, 0), "abnormal": (0, 165, 255)},
    "crawling": {"normal": (255, 0, 0), "abnormal": (0, 0, 255)},
    "crawling_with_weapon": {"normal": (255, 255, 0), "abnormal": (0, 0, 255)},
    "crouching": {"normal": (128, 0, 128), "abnormal": (0, 0, 255)},
    "crouching_with_weapon": {"normal": (0, 255, 255), "abnormal": (0, 0, 255)},
    "cycling": {"normal": (255, 165, 0), "abnormal": (0, 0, 255)},
    "motor_bike": {"normal": (0, 128, 255), "abnormal": (0, 0, 255)},
    "walking": {"normal": (128, 128, 128), "abnormal": (0, 0, 255)},
    "walking_with_weapon": {"normal": (0, 255, 128), "abnormal": (0, 0, 255)}
}

motion_history = deque(maxlen=30)
standing_duration = {}
abnormal_threshold = 10
gathering_threshold = 3
proximity_threshold = 100

motion_threshold = 2000
crowd_density_threshold = 0.3
activity_thresholds = {
    "Creeping": 0.25,
    "crawling": 0.2,
    "crawling_with_weapon": 0.01,
    "crouching": 0.1,
    "crouching_with_weapon": 0.01,
    "cycling": 0.05,
    "motor_bike": 0.3,
    "walking": 0.05,
    "walking_with_weapon": 0.01
}

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://192.168.1.8:8080/h246.sdp")

frame_width = 1280
frame_height = 720
fps = int(cap.get(cv2.CAP_PROP_FPS))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_folder, f"detection_output_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

count = 0
prev_boxes = []
start_time = time.time()

ALERT_SETTING = {
    "running": {"frequency": 2500, "duration": 1000}, 
    "fighting": {"frequency": 3000, "duration": 1500},
    "robbery": {"frequency": 3500, "duration": 2000}, 
    "armed": {"frequency": 4000, "duration": 2000}   
}

last_alert_time = 0
ALERT_COOLDOWN = 2

def trigger_alert(activity_type):
    global last_alert_time
    current_time = time.time()

    if (activity_type in ALERT_SETTING and
        current_time - last_alert_time >= ALERT_COOLDOWN):

        alert = ALERT_SETTING.get(activity_type)
        last_alert_time = current_time

        with open("abnormal_activity_log.txt", "a") as f:
            timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - ALERT: Abnormal {activity_type} detected\n")

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def analyze_crowd_behavior(frame):
    fgMask = backSub.apply(frame)

    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    motion_pixels = cv2.countNonZero(fgMask)
    total_pixels = frame.shape[0] * frame.shape[1]
    crowd_density_c = motion_pixels / total_pixels

    is_crowd = crowd_density_c > crowd_density_threshold

    return is_crowd, fgMask, crowd_density_c

def is_activity_abnormal(activity, crowd_density_c, sudden_movement_s):

    always_abnormal_activity = {"crouching_with_weapon", "crawling_with_weapon", "walking_with_weapon", "Creeping"}
    if activity in always_abnormal_activity:
        return True

    if activity in {"standing", "walking"}:
        return sudden_movement_s

    threshold = activity_thresholds.get(activity, 0.3)

    return crowd_density_c > threshold

def check_gathering(current_boxs):
    centers = []
    for box in current_boxs:
        x1, y1, x2, y2, _ = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers.append(center)

    gathering_groups = []
    checked = set()

    for i, center1 in enumerate(centers):
        if i in checked:
            continue

        group = [i]
        for j, center2 in enumerate(centers):
            if i != j and j not in checked:
                if calculate_distance(center1, center2) > proximity_threshold:
                    group.append(j)

        if len(group) >= gathering_threshold:
            gathering_groups.append(group)
            checked.update(group)

    return gathering_groups

def detect_sudden_movement(prev_boxes_p, current_boxes_c):
    for pre_box, curr_box in zip(prev_boxes_p, current_boxes_c):
        pre_center = ((pre_box[0] + pre_box[2]) // 2, (pre_box[1] + pre_box[3]) // 2)
        curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)

        distance = calculate_distance(pre_center, curr_center)
        if distance > proximity_threshold:
            return True
    return False


heatmap =None
trajectories = defaultdict(list)

abnormal_frames_folder = "abnormal_frames"
if not os.path.exists(abnormal_frames_folder):
    os.makedirs(abnormal_frames_folder)

heatmap_folder = "heatmaps"
if not os.path.exists(heatmap_folder):
    os.makedirs(heatmap_folder)

def save_abnormal_frames(frame, activity):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.join(abnormal_frames_folder, f"abnormal_{activity}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

def save_heatmap(hm, timestamp=None):
    if hm is None:
        print("No heatmap to save")
        return

    heatmap_normalized = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = os.path.join(heatmap_folder, f"heatmap_{timestamp}.jpg")
    cv2.imwrite(filename, heatmap_colored)

def update_heatmap(hm, fg_mask_f):
    if hm is None:
        hm = np.zeros(fg_mask_f.shape, dtype=np.float32)
    hm += fg_mask_f.astype(np.float32)
    return hm

def track_people(prev_gray_p, curr_gray, prev_boxes_p, trajectories_t):
    if prev_gray_p is None or len(prev_boxes_p) == 0:
        return trajectories_t

    # calc optical view
    prev_points = np.array([((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in prev_boxes], dtype=np.float32)
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_p, curr_gray, prev_points, None)

    for i, (new, old) in enumerate(zip(curr_points, prev_points)):
        if status[i]:
            trajectories_t[i].append((int(new[0]), int(new[1])))

    return trajectories_t

prev_gray = None
last_heatmap_time = time.time()
heatmap_interval = 10

while True:
    ret, img = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    is_crowded, fg_mask, crowd_density = analyze_crowd_behavior(img)
    heatmap = update_heatmap(heatmap, fg_mask)

    current_time = time.time()
    if current_time - last_heatmap_time > heatmap_interval:
        save_heatmap(heatmap)
        last_heatmap_time = current_time

    result = model(img)
    current_boxes = []
    abnormal_activities = []
    sudden_movement = detect_sudden_movement(prev_boxes, current_boxes)

    for r in result:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls)
            class_label = class_names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            box_id = f"{box_center[0]}_{box_center[1]}"

            if class_label == 'standing':
                if box_id not in standing_duration:
                    standing_duration[box_id] = current_time
                activity_duration = current_time - standing_duration[box_id]
            else:
                activity_duration = 0
                standing_duration.pop(box_id, None)

            is_abnormal = is_activity_abnormal(class_label, crowd_density, sudden_movement)
            color = colors[class_label]["abnormal" if is_abnormal else "normal"]

            current_boxes.append((x1, y1, x2, y2, class_label))

            status = "ABNORMAL" if is_abnormal else "NORMAL"
            display_label = f"{class_label} - {status}"

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            cv2.putText(
                img,
                display_label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if is_abnormal:
                abnormal_activities.append(class_label)
                save_abnormal_frames(img, class_label)

                if class_label in ALERT_SETTING:
                    trigger_alert(class_label)

        trajectories = track_people(prev_gray, gray, prev_boxes, trajectories)

        prev_boxes = current_boxes.copy()
        prev_gray = gray

        num_human = len(current_boxes)
        density_text = f"Density: {crowd_density:.2f}"
        human_text = f"Human: {num_human}"

        cv2.putText(
            img,
            density_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            human_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        overall_status = "ABNORMAL" if abnormal_activities else "NORMAL"
        status_color = (0, 0, 255) if overall_status == "ABNORMAL" else (0, 255, 0)
        cv2.putText(img, overall_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        out.write(img)
        cv2.imshow("Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if heatmap is not None:
    save_heatmap(heatmap)

out.release()
cap.release()
cv2.destroyAllWindows()

print(f"Video saved successfully at: {output_video_path}")