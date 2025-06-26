import cv2
import mediapipe as mp
import pyautogui

# Initialize
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
screen_w, screen_h = pyautogui.size()

# Nose calibration
calibrated = False
center_x, center_y = 0, 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # 🧭 Use nose tip (landmark 1) for head-based cursor control
        nose = landmarks[1]
        nose_x = int(nose.x * frame_w)
        nose_y = int(nose.y * frame_h)
        cv2.circle(frame, (nose_x, nose_y), 4, (255, 0, 0), -1)

        # 🔧 Calibrate center nose position on first run
        if not calibrated:
            center_x, center_y = nose.x, nose.y
            calibrated = True

        # 🎯 Calculate delta movement from center
        dx = nose.x - center_x
        dy = nose.y - center_y

        # 🎮 Map to screen position with sensitivity factor
        sensitivity = 2.5  # Increase for faster motion
        screen_x = screen_w / 2 + dx * screen_w * sensitivity
        screen_y = screen_h / 2 + dy * screen_h * sensitivity
        pyautogui.moveTo(screen_x, screen_y)

        # 🧿 LEFT eye for cursor (landmarks 145–159)
        left_eye = [landmarks[145], landmarks[159]]
        for point in left_eye:
            x = int(point.x * frame_w)
            y = int(point.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # 🖱️ RIGHT eye blink for click (landmarks 374 & 386)
        right_eye = [landmarks[374], landmarks[386]]
        for point in right_eye:
            x = int(point.x * frame_w)
            y = int(point.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # 🖱️ Blink detection (right eye)
        if (right_eye[0].y - right_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Head & Eye Controlled Mouse', frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break
