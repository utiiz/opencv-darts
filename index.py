import cv2
import numpy as np

# Global list to store points
calibration_points = []


# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        print("Point added at: ", x, y)


def warp_dartboard(frame, points, w=300, h=300):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (w, h))

    return result

# Updating the show_webcam function to incorporate click_event


def show_webcam():
    cap = cv2.VideoCapture("./input/darts.mp4")

    cv2.namedWindow("Webcam Feed")
    cv2.setMouseCallback("Webcam Feed", click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        width, height, _ = frame.shape

        for point in calibration_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            if len(calibration_points) >= 4:
                frame = warp_dartboard(
                    frame, calibration_points, width, height)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


show_webcam()
