import os
import sys
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import cv2
import time
import json
from playsound import playsound
import mediapipe as mp

# Constants
EAR_CONSEC_FRAMES = 3
SETTINGS_FILE = "settings.json"
SOUND_FILE = "resources\\beep.mp3"

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

DEFAULT_SETTINGS = {
        "ear_threshold": 0.2,
        "alert_time": 5,
        "show_video": True
    }


def resource_path(relative_path: str):
    """ Get absolute path to resource, works for dev and for PyInstaller.
     To correctly search for resources when building the .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_SETTINGS


def save_settings(settings: dict):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)


def calculate_ear(eye_points):
    """Eye Aspect Ratio calculation"""
    A = abs(eye_points[1][1] - eye_points[5][1])  # vertical points
    B = abs(eye_points[2][1] - eye_points[4][1])
    C = abs(eye_points[0][0] - eye_points[3][0])  # horizontal point
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear


def get_eye_coords(indices, landmarks, width, height):
    return [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices]


class BlinkMonitorApp:
    def __init__(self, root):
        self.video_thread = None
        self.root = root
        self.root.title("Blink tracker")
        self.root.geometry("600x380")
        self.root.resizable(True, True)

        # Load settings
        settings = load_settings()

        self.EAR_THRESHOLD = settings.get("ear_threshold", 0.19)
        self.ALERT_TIME = settings.get("alert_time", 5)
        self.SHOW_VIDEO = settings.get("show_video", True)

        # Handling window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_label = tk.Label(root, text="Status: Waiting for camera...", font=("Arial", 14))
        self.status_label.pack(pady=20)

        self.timer_label = tk.Label(root, text="Time since last blink: 0 sec", font=("Arial", 12))
        self.timer_label.pack(pady=10)

        self.blink_count_label = tk.Label(root, text="Number of blinks: 0", font=("Arial", 12))
        self.blink_count_label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start the camera", command=self.start_monitoring)
        self.start_button.pack(pady=10)

        # Variable for video display
        self.show_video = tk.BooleanVar(value=True)

        # Checkbox to show video.
        self.video_checkbox = tk.Checkbutton(
            root,
            text="Show monitor",
            variable=self.show_video,
            font=("Arial", 12)
        )
        self.video_checkbox.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop the camera", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.threshold_label = tk.Label(root, text="Sensitivity", font=("Arial", 12))
        self.threshold_label.pack(side=tk.LEFT, pady=5, padx=2)
        self.ear_entry = tk.Entry(root, width=10, font=("Arial", 12))
        self.ear_entry.insert(0, str(self.EAR_THRESHOLD))
        self.ear_entry.pack(side=tk.LEFT, padx=5)

        self.alert_time_label = tk.Label(root, text="Time between blinks, sec.", font=("Arial", 12))
        self.alert_time_label.pack(side=tk.LEFT, pady=10, padx=2)
        self.alert_time_entry = tk.Entry(root, width=10, font=("Arial", 12))
        self.alert_time_entry.insert(0, str(self.ALERT_TIME))
        self.alert_time_entry.pack(side=tk.LEFT, padx=5)

        self.running = False
        self.last_blink_time = time.time()
        self.last_alert_time = time.time()

    def on_close(self):
        if self.running:
            self.running = False  # Stopping the video loop
            self.root.update_idletasks()

        # Wait for the stream to complete, but no more than 1 second
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=5.0)

        # Make sure that the OpenCV window is closed
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # It is now safe to close the GUI
        self.root.destroy()

    def save_current_settings(self):
        settings = {
            "ear_threshold": self.EAR_THRESHOLD,
            "alert_time": self.ALERT_TIME,
            "show_video": self.show_video.get()
        }
        save_settings(settings)

    def update_params(self):
        verified = True
        try:
            value = float(self.ear_entry.get())
            if 0.0 <= value <= 5.0:
                self.EAR_THRESHOLD = value
            else:
                raise ValueError
        except Exception as e:
            messagebox.showwarning("Error", "Enter a number between 0.0 and 5.0")
            verified = False

        if verified:
            try:
                alert_time = float(self.alert_time_entry.get())
                if 1.0 <= alert_time <= 60.0:
                    self.ALERT_TIME = alert_time
                else:
                    raise ValueError
            except Exception as e:
                messagebox.showwarning("Error", "Enter a number between 1.0 and 60.0")
                verified = False
        self.save_current_settings()
        return verified

    def start_monitoring(self):
        self.running = True

        if self.update_params() is False:
            self.running = False
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Tracking active...")
        # Save the stream reference
        self.video_thread = Thread(target=self.run_monitoring, daemon=True)
        self.video_thread.start()

    def stop_monitoring(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Tracking stopped")

    def draw_info(self, frame, ear):
        # Text to be displayed on the frame
        text_ear = f"EAR: {round(ear, 3)}"

        # Text positions
        pos_ear = (10, 20)

        # Color and text options
        color = (100, 100, 255) if ear < self.EAR_THRESHOLD else (100, 255, 100)
        thickness = 1
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw text on the frame
        cv2.putText(frame, text_ear, pos_ear, font, font_scale, color, thickness)

    def run_monitoring(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error," "Failed to connect to camera.")
            self.stop_monitoring()
            return

        blink_count = 0
        close_eyes = False
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_time = time.time()

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye = get_eye_coords(LEFT_EYE_INDICES, landmarks, w, h)
                right_eye = get_eye_coords(RIGHT_EYE_INDICES, landmarks, w, h)

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < self.EAR_THRESHOLD:
                    self.last_blink_time = current_time  # update the last blink time
                    if close_eyes is False:
                        blink_count += 1
                        self.blink_count_label.config(text=f"Number of blinks: {blink_count}")
                        close_eyes = True
                else:
                    close_eyes = False

                blink_duration = current_time - self.last_blink_time
                if blink_duration > self.ALERT_TIME:
                    self.status_label.config(text="❗ Too long without blinking!")
                    try:
                        playsound(resource_path(SOUND_FILE))
                    except Exception as e:
                        print("⚠️ Failed to play the sound:", e)
                    self.last_alert_time = current_time
                    self.last_blink_time = current_time  # update the last blink time
                    self.status_label.config(text="👁 Eyes open")

                elapsed = round(current_time - self.last_blink_time, 1)
                self.timer_label.config(text=f"⏱ Time since last blink: {elapsed} sec")

                # Eye visualization
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 1, (0, 255, 0), -1)

                self.draw_info(frame=frame, ear=ear)
            else:
                self.status_label.config(text="⚠️ No face was found")
                self.timer_label.config(text="")

            if self.show_video.get():
                # Display frame in a separate window
                cv2.imshow('Blink Monitor', frame)
                cv2.waitKey(1)

        # Freeing up resources
        self.cap.release()
        if self.show_video.get():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = BlinkMonitorApp(root)
    root.mainloop()

