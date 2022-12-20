import cv2
import mediapipe as mp
import time

start_time = time.time()
mp_drawing = mp.solutions.drawing_utils             # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles     # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                         # mediapipe 姿勢偵測
# 影片
# cap = cv2.VideoCapture("/home/ezio/openpose/examples/video/cut.mp4")
cap = cv2.VideoCapture("video/IMG_2319.MOV")
# 相機
# cap = cv2.VideoCapture(0)


fps = cap.get(cv2.CAP_PROP_FPS)  # 視頻平均幀率
print("fps", fps)
frame_width = 1280
frame_height = 720
counter = 0

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        if cv2.waitKey(5) == ord('q'):
            break  # 按下 q 鍵停止

        counter += 1  # 計算幀數

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if not results.pose_landmarks:
            continue
        print(
            f'WRIST coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height})'
        )

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if (time.time() - start_time) != 0:
            cv2.putText(image, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        3)
            resize = cv2.resize(image, (frame_width, frame_height))
            cv2.imshow('MediaPipe Pose', resize)
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
        # time.sleep(1 / fps)  # 按原幀率播放

cap.release()
cv2.destroyAllWindows()
