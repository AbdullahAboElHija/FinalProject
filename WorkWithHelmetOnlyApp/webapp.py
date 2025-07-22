import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sort import Sort
import requests
import smtplib
from email.message import EmailMessage
import ssl
import collections

st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🚧", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {background-color: #FFFFFF; color: #333333;}
    [data-testid="stSidebar"] {background-color: #808080; color: #FFFFFF; padding: 10px;}
    h1, h2, h3 {color: #FFA500; font-family: 'Arial', sans-serif; font-weight: bold; text-shadow: 1px 1px 2px #333333;}
    .stButton>button {background-color: #FFA500; color: #FFFFFF; border: 2px solid #333333; border-radius: 5px; font-weight: bold; box-shadow: 2px 2px 4px #808080;}
    .stButton>button:hover {background-color: #FFFF00; color: #333333; box-shadow: 4px 4px 8px #808080;}
    .stTabs [data-baseweb="tab"] {background-color: #808080; color: #FFFFFF; font-weight: bold; border-radius: 5px 5px 0 0;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #FFA500; color: #FFFFFF;}
    .image-frame {border: 4px solid #FFA500; border-radius: 10px; padding: 5px; background-color: #808080; box-shadow: 3px 3px 6px #333333;}
    .username {text-align: center; color: #FFFF00; font-weight: bold; margin-top: 5px; text-shadow: 1px 1px 2px #333333;}
    .stDataFrame {border: 2px solid #FFA500; border-radius: 5px;}
    .highlight-container {border: 2px solid #FFA500; border-radius: 10px; padding: 10px; background-color: #F5F5F5; box-shadow: 2px 2px 5px #808080;}
    </style>
""", unsafe_allow_html=True)

DEFAULT_IMAGE_URL = "https://cdn-icons-png.flaticon.com/512/9131/9131478.png"
webcam_csv_file = 'webcam_ppe_tracking.csv'
video_csv_file = 'video_ppe_tracking.csv'

# Initialize CSV files if they don't exist
for file in [webcam_csv_file, video_csv_file]:
    if not os.path.exists(file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(file, index=False)

# Session state initialization
if "receiver_email" not in st.session_state:
    st.session_state.receiver_email = "workwithhelmtonly@gmail.com"
if "profile_image" not in st.session_state:
    if not os.path.exists("default_profile.jpg"):
        response = requests.get(DEFAULT_IMAGE_URL)
        with open("default_profile.jpg", "wb") as f:
            f.write(response.content)
    st.session_state.profile_image = "default_profile.jpg"
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "video_active" not in st.session_state:
    st.session_state.video_active = False
if "session_data" not in st.session_state:
    st.session_state.session_data = {}
if "username" not in st.session_state:
    st.session_state.username = "SafetyInspector"
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO('ppe.pt')
    st.session_state.tracker = Sort()
# New session states for violation tracking
if "violation_start_time" not in st.session_state:
    st.session_state.violation_start_time = None
if "violation_triggered" not in st.session_state:
    st.session_state.violation_triggered = False
if "violation_evidence_frame" not in st.session_state:
    st.session_state.violation_evidence_frame = None
# Add a session state variable to track the start of the no-violation period
if "no_violation_start_time" not in st.session_state:
    st.session_state.no_violation_start_time = None
# Add a session state variable for person tracks (for stable ID assignment)
if "person_tracks" not in st.session_state:
    st.session_state.person_tracks = {}
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
# Add a session state variable for per-person, per-item violation tracking
if "violation_tracking" not in st.session_state:
    # Structure: {person_id: {item: {"start_time": float, "logged": bool}}}
    st.session_state.violation_tracking = {}

# --- User selection for which classes to monitor ---
# Map user-friendly names to YOLO class names
SAFETY_CLASS_MAP = {
    "Hardhat": ["Hardhat", "NO-Hardhat"],
    "Safety Vest": ["Safety Vest", "NO-Safety Vest"],
    "Mask": ["Mask", "NO-Mask"]
}

# Helper functions for appearance and position matching
def compute_histogram(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    person_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if person_img.size == 0:
        return None
    hist = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def iou(boxA, boxB):
    # Compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def match_track(tracks, bbox, hist, iou_thresh=0.3, hist_thresh=0.5):
    best_id = None
    best_score = 0
    for tid, tinfo in tracks.items():
        iou_score = iou(bbox, tinfo['bbox'])
        if tinfo['hist'] is not None and hist is not None:
            hist_score = cv2.compareHist(tinfo['hist'], hist, cv2.HISTCMP_CORREL)
        else:
            hist_score = 0
        # Both IoU and histogram must be above threshold
        if iou_score > iou_thresh and hist_score > hist_thresh:
            score = iou_score + hist_score
            if score > best_score:
                best_score = score
                best_id = tid
    return best_id

def send_email_with_attachment(receiver_email, csv_file, image_file):
    sender_email = "workwithhelmtonly@gmail.com"
    sender_password = "xddnathbrnyaefwg"  # Use App Password for Gmail
    subject = "Safety Monitoring Report"
    body = "Please find attached the safety report and violation snapshot."

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with open(csv_file, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="csv", filename=os.path.basename(csv_file))
        with open(image_file, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_file))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.success(f"Email sent to {receiver_email}")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")

def save_session_data(csv_file):
    # Only log violations that have been continuously detected for >= 3 seconds and not already logged in this session
    if not st.session_state.session_data:
        return
    if not os.path.exists(csv_file):
        pd.DataFrame(columns=["Timestamp", "Person ID", "Equipment Worn", "Equipment Not Worn"]).to_csv(csv_file, index=False)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, 'a', newline='') as file:
        for obj_id, equipment in st.session_state.session_data.items():
            # Only log if there is a violation that has been marked as logged
            not_worn_to_log = []
            if obj_id in st.session_state.violation_tracking:
                for item, vinfo in st.session_state.violation_tracking[obj_id].items():
                    if vinfo["logged"]:
                        not_worn_to_log.append(item)
                        # Mark as not to log again until violation ends
                        st.session_state.violation_tracking[obj_id][item]["logged"] = False
            if not_worn_to_log:
                worn = '+'.join(equipment['worn']) or "None"
                not_worn = '+'.join(not_worn_to_log) or "None"
                file.write(f"{timestamp},{obj_id},{worn},{not_worn}\n")

def process_frame(frame):
    model = st.session_state.yolo_model
    tracker = st.session_state.tracker
    allowed_labels = st.session_state.allowed_labels
    violation_in_this_frame = False
    current_time = time.time()
    violation_timer = st.session_state.violation_timer  # Use user-selected timer for both logging and email

    results = model(frame, device=0) 
    
    detections = []
    person_bboxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu()
            conf = box.conf.item()
            label = model.names[int(box.cls.item())]
            if conf >= 0.4 and label == "Person":
                detections.append([x1, y1, x2, y2, conf])
                person_bboxes.append([x1, y1, x2, y2])

    tracked_objects = tracker.update(np.array(detections))
    st.session_state.frame_count += 1
    new_tracks = {}
    used_ids = set()
    # Prepare a dict to store current frame's not_worn/worn for each person
    current_frame_status = {}
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        bbox = [x1, y1, x2, y2]
        hist = compute_histogram(frame, bbox)
        match_id = match_track(st.session_state.person_tracks, bbox, hist)
        if match_id is not None and match_id not in used_ids:
            stable_id = match_id
        else:
            stable_id = obj_id
        used_ids.add(stable_id)
        new_tracks[stable_id] = {'bbox': bbox, 'hist': hist, 'last_seen': st.session_state.frame_count}
        if stable_id not in st.session_state.session_data:
            st.session_state.session_data[stable_id] = {"worn": set(), "not_worn": set()}
        # Track current frame's status for this person
        current_frame_status[stable_id] = {"worn": set(), "not_worn": set()}
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls.item())]
                conf = box.conf.item()
                if conf >= 0.4 and label in allowed_labels:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu()
                    if x1 <= bx1 <= x2 and y1 <= by1 <= y2:
                        if label.startswith("NO-"):
                            current_frame_status[stable_id]["not_worn"].add(label)
                            violation_in_this_frame = True
                        else:
                            current_frame_status[stable_id]["worn"].add(label)
                    color = (0, 255, 0) if "NO-" not in label else (255, 0, 0)
                    cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
                    cv2.putText(frame, label, (int(bx1), int(by1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {stable_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Clean up old tracks (not seen for 30 frames)
    st.session_state.person_tracks = {tid: tinfo for tid, tinfo in new_tracks.items() if st.session_state.frame_count - tinfo['last_seen'] < 30}

    # --- Violation time tracking logic ---
    # For each person, for each item, update violation_tracking
    for pid, status in current_frame_status.items():
        if pid not in st.session_state.violation_tracking:
            st.session_state.violation_tracking[pid] = {}
        # Handle not_worn items
        for item in status["not_worn"]:
            if item not in st.session_state.violation_tracking[pid]:
                st.session_state.violation_tracking[pid][item] = {"start_time": current_time, "logged": False}
            else:
                # If not logged and violation_timer seconds passed, mark as logged
                if not st.session_state.violation_tracking[pid][item]["logged"]:
                    if current_time - st.session_state.violation_tracking[pid][item]["start_time"] >= violation_timer:
                        st.session_state.violation_tracking[pid][item]["logged"] = True
        # Handle worn items (reset tracking if item is now worn)
        for item in list(st.session_state.violation_tracking[pid].keys()):
            if item not in status["not_worn"]:
                st.session_state.violation_tracking[pid].pop(item)
    # Clean up tracking for people not detected anymore
    for pid in list(st.session_state.violation_tracking.keys()):
        if pid not in current_frame_status:
            st.session_state.violation_tracking.pop(pid)
    # Update session_data for display (not for logging)
    for pid, status in current_frame_status.items():
        st.session_state.session_data[pid]["worn"] = status["worn"]
        st.session_state.session_data[pid]["not_worn"] = status["not_worn"]
    st.session_state.last_frame = frame
    return frame, violation_in_this_frame

with st.sidebar:
    st.image(st.session_state.profile_image, width=100)
    # Add multiselect for safety items
    selected_items = st.multiselect(
        "Monitor for:",
        ["Hardhat", "Safety Vest", "Mask"],
        default=["Hardhat", "Safety Vest", "Mask"]
    )
    # Build allowed_labels set based on user selection
    st.session_state.allowed_labels = set()
    for item in selected_items:
        st.session_state.allowed_labels.update(SAFETY_CLASS_MAP[item])
    # Add a single timer input for both detection and email
    st.session_state.violation_timer = st.number_input(
        "Violation timer (seconds)", min_value=1, max_value=10, value=3, step=1,
        help="How long a violation must persist before it is logged and an email is sent."
    )
    page_options = {
        "🏠 Dashboard": "Dashboard",
        "✉️ Set Receiver Email": "Set Receiver Email",
        "📈 Analytics": "Analytics",
        "👷 About Me": "About Me"
    }
    page = st.radio("Navigation", list(page_options.keys()), format_func=lambda x: x)
    selected_page = page_options[page]

if selected_page == "Dashboard":
    st.title("🚧 Construction Safety Dashboard")
    tab1, tab2 = st.tabs(["📹 Webcam", "📼 Video Upload"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fps = st.slider("Webcam FPS", 1, 30, 15, key="webcam_fps")
            if st.button("▶️ Start Webcam", key="start_webcam"):
                st.session_state.webcam_active = True
                st.session_state.session_data = {}
                st.session_state.violation_start_time = None
                st.session_state.violation_triggered = False
                st.session_state.violation_evidence_frame = None

            if st.button("⏹️ Stop Webcam", key="stop_webcam"):
                st.session_state.webcam_active = False
                # Only update CSV and send email if a violation was triggered and not cleared by the 3-second rule
                if st.session_state.violation_triggered:
                    save_session_data(webcam_csv_file)
                    st.warning(f"Violation detected for over {st.session_state.violation_timer} seconds. Sending email report.")
                    evidence_frame_to_save = st.session_state.violation_evidence_frame if st.session_state.violation_evidence_frame is not None else st.session_state.last_frame
                    if evidence_frame_to_save is not None:
                        evidence_image = "violation_snapshot.jpg"
                        cv2.imwrite(evidence_image, evidence_frame_to_save)
                        send_email_with_attachment(st.session_state.receiver_email, webcam_csv_file, evidence_image)
                    else:
                        st.error("Could not send email. No evidence frame was captured.")
                else:
                    st.info(f"Processing stopped. No violations were sustained for the required duration ({st.session_state.violation_timer} seconds) or were cleared after 3 seconds of no violation.")

            if st.session_state.webcam_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not open webcam.")
                    st.session_state.webcam_active = False
                else:
                    placeholder = st.empty()
                    try:
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Error: Could not read frame from webcam.")
                                break
                            
                            processed_frame, violation_detected = process_frame(frame)

                            if violation_detected:
                                # Violation detected, reset no_violation_start_time
                                st.session_state.no_violation_start_time = None
                                if st.session_state.violation_start_time is None:
                                    st.session_state.violation_start_time = time.time()
                                elif time.time() - st.session_state.violation_start_time >= st.session_state.violation_timer:
                                    st.session_state.violation_triggered = True
                                    if st.session_state.violation_evidence_frame is None:
                                        st.session_state.violation_evidence_frame = processed_frame
                            else:
                                # No violation detected
                                st.session_state.violation_start_time = None
                                if st.session_state.no_violation_start_time is None:
                                    st.session_state.no_violation_start_time = time.time()
                                elif time.time() - st.session_state.no_violation_start_time >= 3:
                                    # No violation for 3 seconds: reset violation states and skip CSV/email
                                    st.session_state.violation_triggered = False
                                    st.session_state.violation_evidence_frame = None
                                    st.session_state.session_data = {}
                                    st.session_state.no_violation_start_time = None
                                    continue  # Skip the rest of the loop

                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            placeholder.image(frame_rgb, caption=st.session_state.username, use_column_width=True)
                            time.sleep(1/fps)
                    except Exception as e:
                        st.error(f"Webcam error: {str(e)}")
                    finally:
                        cap.release()
                        st.session_state.webcam_active = False

        with col2:
            with st.container():
                st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
                st.write("### 🔔 Detected Classes (Webcam)")
                detected_classes = set()
                for equipment in st.session_state.session_data.values():
                    detected_classes.update(equipment['worn'])
                    detected_classes.update(equipment['not_worn'])
                st.write(detected_classes)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            fps = st.slider("Video FPS", 1, 30, 15, key="video_fps")
            uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_file:
                if st.button("▶️ Process Video", key="start_video"):
                    st.session_state.video_active = True
                    st.session_state.session_data = {}
                    st.session_state.violation_start_time = None
                    st.session_state.violation_triggered = False
                    st.session_state.violation_evidence_frame = None

                if st.button("⏹️ Stop Processing", key="stop_video"):
                    st.session_state.video_active = False
                    # Only update CSV and send email if a violation was triggered and not cleared by the 3-second rule
                    if st.session_state.violation_triggered:
                        save_session_data(video_csv_file)
                        st.warning(f"Violation detected for over {st.session_state.violation_timer} seconds. Sending email report.")
                        evidence_frame_to_save = st.session_state.violation_evidence_frame if st.session_state.violation_evidence_frame is not None else st.session_state.last_frame
                        if evidence_frame_to_save is not None:
                            evidence_image = "violation_snapshot.jpg"
                            cv2.imwrite(evidence_image, evidence_frame_to_save)
                            send_email_with_attachment(st.session_state.receiver_email, video_csv_file, evidence_image)
                        else:
                            st.error("Could not send email. No evidence frame was captured.")
                    else:
                        st.info(f"Processing stopped. No violations were sustained for the required duration ({st.session_state.violation_timer} seconds) or were cleared after 3 seconds of no violation.")

                if st.session_state.video_active:
                    # FIX 2: Properly handle the temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    tfile.close()

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video file.")
                        st.session_state.video_active = False
                    else:
                        placeholder = st.empty()
                        try:
                            while cap.isOpened() and st.session_state.video_active:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                processed_frame, violation_detected = process_frame(frame)
                                
                                if violation_detected:
                                    # Violation detected, reset no_violation_start_time
                                    st.session_state.no_violation_start_time = None
                                    if st.session_state.violation_start_time is None:
                                        st.session_state.violation_start_time = time.time()
                                    elif time.time() - st.session_state.violation_start_time >= st.session_state.violation_timer:
                                        st.session_state.violation_triggered = True
                                        if st.session_state.violation_evidence_frame is None:
                                            st.session_state.violation_evidence_frame = processed_frame
                                else:
                                    # No violation detected
                                    st.session_state.violation_start_time = None
                                    if st.session_state.no_violation_start_time is None:
                                        st.session_state.no_violation_start_time = time.time()
                                    elif time.time() - st.session_state.no_violation_start_time >= 3:
                                        # No violation for 3 seconds: reset violation states and skip CSV/email
                                        st.session_state.violation_triggered = False
                                        st.session_state.violation_evidence_frame = None
                                        st.session_state.session_data = {}
                                        st.session_state.no_violation_start_time = None
                                        continue  # Skip the rest of the loop

                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                placeholder.image(frame_rgb, caption=st.session_state.username, use_column_width=True)
                                time.sleep(1/fps)
                        except Exception as e:
                            st.error(f"Video processing error: {str(e)}")
                        finally:
                            cap.release()
                            os.unlink(video_path) # Delete the file using its path
                            st.session_state.video_active = False

        with col2:
            with st.container():
                st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
                st.write("### 🔔 Detected Classes (Video)")
                detected_classes = set()
                for equipment in st.session_state.session_data.values():
                    detected_classes.update(equipment['worn'])
                    detected_classes.update(equipment['not_worn'])
                st.write(detected_classes)
                st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "Set Receiver Email":
    st.title("✉️ Set Receiver Email")
    with st.container():
        st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
        email = st.text_input("Receiver Email", st.session_state.receiver_email)
        if st.button("💾 Save"):
            st.session_state.receiver_email = email
            st.success("Receiver email updated!")
        st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "Analytics":
    st.title("📈 Safety Analytics")
    tab1, tab2 = st.tabs(["📹 Webcam Data", "📼 Video Data"])
    
    def plot_violations_by_person(df, title, container):
        if df.empty:
            container.warning(f"No data available for {title}.")
            return
        violation_summary = {}
        for index, row in df.iterrows():
            person_id = row["Person ID"]
            not_worn_list = str(row["Equipment Not Worn"]).split()
            if person_id not in violation_summary:
                violation_summary[person_id] = {"NO-Hardhat": 0, "NO-Mask": 0, "NO-Safety Vest": 0}
            for item in not_worn_list:
                item = item.strip()
                if item in violation_summary[person_id]:
                    violation_summary[person_id][item] += 1
        violation_df = pd.DataFrame.from_dict(violation_summary, orient="index").reset_index()
        violation_df = violation_df.rename(columns={"index": "Person ID"})
        violation_df = violation_df.fillna(0)
        long_df = violation_df.melt(id_vars=["Person ID"], var_name="Violation Type", value_name="Count")
        container.write(f"### 📊 {title} - Violations by Person ID")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.barplot(x="Person ID", y="Count", hue="Violation Type", data=long_df, ax=ax, 
                   palette={"NO-Hardhat": "#FFA500", "NO-Mask": "#FFFF00", "NO-Safety Vest": "#808080"})
        ax.set_title(f"{title} - Equipment Not Worn by Person ID", color="#FFA500")
        ax.set_xlabel("Person ID", color="#333333")
        ax.set_ylabel("Violation Count", color="#333333")
        plt.xticks(rotation=45, color="#333333")
        plt.yticks(color="#333333")
        ax.spines['top'].set_color('#808080')
        ax.spines['right'].set_color('#808080')
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        plt.tight_layout()
        container.pyplot(fig)
        plt.close()

    def plot_safety_check_summary(df, title, container):
        if df.empty:
            container.warning(f"No data available for {title}.")
            return
        safety_check = {}
        for index, row in df.iterrows():
            person_id = row["Person ID"]
            not_worn_list = str(row["Equipment Not Worn"]).split()
            if person_id not in safety_check:
                safety_check[person_id] = "Pass"
            for item in not_worn_list:
                if item.startswith("NO-"):
                    safety_check[person_id] = "Fail"
                    break
        safety_df = pd.DataFrame(list(safety_check.items()), columns=["Person ID", "Safety Check"])
        container.write(f"### ✅ {title} - Safety Check Summary")
        container.dataframe(safety_df)
        summary_count = safety_df["Safety Check"].value_counts()
        container.write(f"### 📊 {title} - Pass/Fail Summary")
        fig, ax = plt.subplots()
        sns.set_style("whitegrid")
        sns.barplot(x=summary_count.index, y=summary_count.values, hue=summary_count.index, 
                   palette={"Pass": "#FFFF00", "Fail": "#FFA500"}, ax=ax, legend=False)
        ax.set_ylabel("Count", color="#333333")
        ax.set_title(f"{title} - Safety Check Result", color="#FFA500")
        plt.xticks(color="#333333")
        plt.yticks(color="#333333")
        ax.spines['top'].set_color('#808080')
        ax.spines['right'].set_color('#808080')
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        plt.tight_layout()
        container.pyplot(fig)
        plt.close()

    with tab1:
        with st.container():
            st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
            if os.path.exists(webcam_csv_file):
                df = pd.read_csv(webcam_csv_file)
                df["Person ID"] = df["Person ID"].astype(str)

                # --- PPE Compliance Bar Chart (Webcam Data) ---
                ppe_items = ["Hardhat", "Mask", "Safety Vest"]
                compliance = []
                for _, row in df.iterrows():
                    person_id = str(row["Person ID"])
                    worn = set(str(row["Equipment Worn"]).split('+'))
                    not_worn = set(str(row["Equipment Not Worn"]).split('+'))
                    person_result = {"Person ID": person_id}
                    for item in ppe_items:
                        if item in worn:
                            person_result[item] = 1
                        elif f"NO-{item}" in not_worn:
                            person_result[item] = 0
                        else:
                            person_result[item] = 0
                    compliance.append(person_result)
                if compliance:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    compliance_df = pd.DataFrame(compliance)
                    x = np.arange(len(compliance_df["Person ID"]))
                    width = 0.2
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Assign a unique color for each PPE item
                    item_colors = {
                        "Hardhat": ("#FFA500", "#FFE5B4"),      # orange, light orange
                        "Mask": ("#1E90FF", "#B0C4DE"),         # blue, light blue/gray
                        "Safety Vest": ("#32CD32", "#C1E1C1")  # green, light green
                    }
                    for idx, item in enumerate(ppe_items):
                        values = compliance_df[item]
                        bar_colors = [item_colors[item][0] if v == 1 else item_colors[item][1] for v in values]
                        ax.bar(x + (idx - 1) * width, values, width, label=item, color=bar_colors)
                    ax.set_ylabel("Worn (1) / Not Worn (0)")
                    ax.set_xlabel("Person ID")
                    ax.set_title("PPE Compliance by Person ID")
                    ax.set_xticks(x)
                    ax.set_xticklabels(compliance_df["Person ID"])
                    ax.set_ylim(-0.1, 1.1)
                    ax.legend(title="PPE Item")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No compliance data to display.")

                plot_violations_by_person(df, "Webcam Data", tab1)
                plot_safety_check_summary(df, "Webcam Data", tab1)
                tab1.write("### 📋 Webcam Data - Raw Data")
                tab1.dataframe(df)
            else:
                tab1.warning("No webcam data found. Please run detection first.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        with st.container():
            st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
            if os.path.exists(video_csv_file):
                df = pd.read_csv(video_csv_file)
                df["Person ID"] = df["Person ID"].astype(str)

                # --- PPE Compliance Bar Chart (Video Data) ---
                ppe_items = ["Hardhat", "Mask", "Safety Vest"]
                compliance = []
                for _, row in df.iterrows():
                    person_id = str(row["Person ID"])
                    worn = set(str(row["Equipment Worn"]).split('+'))
                    not_worn = set(str(row["Equipment Not Worn"]).split('+'))
                    person_result = {"Person ID": person_id}
                    for item in ppe_items:
                        if item in worn:
                            person_result[item] = 1
                        elif f"NO-{item}" in not_worn:
                            person_result[item] = 0
                        else:
                            person_result[item] = 0
                    compliance.append(person_result)
                if compliance:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    compliance_df = pd.DataFrame(compliance)
                    x = np.arange(len(compliance_df["Person ID"]))
                    width = 0.2
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Assign a unique color for each PPE item
                    item_colors = {
                        "Hardhat": ("#FFA500", "#FFA500"),      # orange, light orange
                        "Mask": ("#1E90FF", "#B0C4DE"),         # blue, light blue/gray
                        "Safety Vest": ("#32CD32", "#C1E1C1")  # green, light green
                    }
                    for idx, item in enumerate(ppe_items):
                        values = compliance_df[item]
                        bar_colors = [item_colors[item][0] if v == 1 else item_colors[item][1] for v in values]
                        ax.bar(x + (idx - 1) * width, values, width, label=item, color=bar_colors)
                    ax.set_ylabel("Worn (1) / Not Worn (0)")
                    ax.set_xlabel("Person ID")
                    ax.set_title("PPE Compliance by Person ID")
                    ax.set_xticks(x)
                    ax.set_xticklabels(compliance_df["Person ID"])
                    ax.set_ylim(-0.1, 1.1)
                    ax.legend(title="PPE Item")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No compliance data to display.")

                plot_violations_by_person(df, "Video Data", tab2)
                plot_safety_check_summary(df, "Video Data", tab2)
                tab2.write("### 📋 Video Data - Raw Data")
                tab2.dataframe(df)
            else:
                tab2.warning("No video data found. Please run detection first.")
            st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "About Me":
    st.title("👷 About Me")
    with st.container():
        st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
        username = st.text_input("Username", st.session_state.username)
        if st.button("💾 Update Username"):
            st.session_state.username = username
            st.success("Username updated!")
        st.selectbox("Gender", ["Male", "Female", "Other"])
        uploaded_profile = st.file_uploader("📸 Upload Profile Image")
