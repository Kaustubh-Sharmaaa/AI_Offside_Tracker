import cv2
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans
import numpy as np

def open_video(time):
    input_video = "videos/sample_clip.mp4"
    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = time * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    return frame

def detection(frame):
    model = YOLO("models/yolov8n.pt")
    results = model(frame)
    boxes = results[0].boxes
    return boxes

def team_detection(frame, boxes):
    player_teams = []
    hue_list = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0]
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        torso_crop = frame[y : y + int(h / 2), x : x + w]
        if torso_crop.size == 0:
            continue

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        hue_values = hsv[:, :, 0].flatten()
        average_hue = np.mean(hue_values)

        hue_list.append(average_hue)
        bbox = (x, y, w, h)
        player_teams.append([i, average_hue, bbox])

    # Cluster players into 2 teams
    kmeans = KMeans(n_clusters=2, n_init=10).fit(np.array(hue_list).reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for idx, label in enumerate(labels):
        player_teams[idx].append(label)  # [id, hue, bbox, team_label]

    return player_teams, centers

def draw_labels(frame, player_teams, centers):
    team_colors = {0: (255, 0, 0), 1: (0, 0, 255)}  # Blue & Red
    referee_color = (0, 255, 255)  # Yellow

    center_0 = centers[0][0]
    center_1 = centers[1][0]
    hue_gap = abs(center_0 - center_1)
    threshold = hue_gap * 0.4  # Adaptive threshold

    for player in player_teams:
        player_id, hue, (x, y, w, h), team_label = player
        dist0 = abs(hue - center_0)
        dist1 = abs(hue - center_1)

        if dist0 > threshold and dist1 > threshold:
            label = "Referee"
            color = referee_color
            player.append("referee")
        else:
            label = f"Team {team_label}"
            color = team_colors[team_label]
            player.append("player")

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def detect_offside(frame, player_teams, attacking_team):
    defending_team = 1 - attacking_team
    attackers = []
    defenders = []

    for player in player_teams:
        _, _, (x, y, w, h), team_label, role = player
        if role != "player":
            continue
        if team_label == attacking_team:
            attackers.append((x, y, w, h))
        else:
            defenders.append((x, y, w, h))

    if len(defenders) < 2:
        print("⚠️ Not enough defenders to determine offside line.")
        return frame

    # Sort defenders by x-position (right to left)
    defenders_sorted = sorted(defenders, key=lambda b: b[0], reverse=True)
    last_defender_x = defenders_sorted[0][0]  # Right-most (deepest) defender



    # Draw the offside line
    height = frame.shape[0]
    cv2.line(frame, (last_defender_x, 0), (last_defender_x, height), (0, 255, 0), 2)

    # Check each attacker
    for (x, y, w, h) in attackers:
        if x > last_defender_x:
            label = "Offside"
            color = (0, 0, 0)  # Red
        else:
            label = ""
            color = (0 ,255, 255)

        cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

# ========== MAIN ==========
frame = None
if not frame:
    time = float(input("Enter the second to extract frame from: "))
    frame = open_video(time)
    boxes = detection(frame)
    player_teams, centers = team_detection(frame, boxes)
    labeled_frame = draw_labels(frame, player_teams, centers)
    cv2.imwrite("outputs/team_labeled.jpg", labeled_frame)

    attacking_team = int(input("Which team is attacking? (0 or 1): "))
    offside_frame = detect_offside(labeled_frame, player_teams, attacking_team)
    cv2.imwrite("outputs/offside_result.jpg", offside_frame)
    print("✅ Offside analysis saved as outputs/offside_result.jpg")
