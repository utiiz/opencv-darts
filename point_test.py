import cv2
import numpy as np
import math

# Global variable to store points
points = []
img = None
image = None


def capture_image():
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Video Stream', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture the frame
            cv2.imwrite('dartboard.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()


def click_event(event, x, y, flags, params):
    global img
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        cv2.imshow('image', img)
        if len(points) == 4:
            image = draw_board(img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def calculate_point(center, radius, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x = int(center[0] + radius * math.cos(angle_radians))
    # Subtracting because y-coordinates increase downwards
    y = int(center[1] - radius * math.sin(angle_radians))
    return (x, y)


def draw_board(image):
    # Assuming the dartboard image is 800x800 pixels, and the board's actual size is centered and fills the image
    img_size = 800
    board_radius = img_size // 2
    center = (board_radius, board_radius)

    board_diameter_pixels = img_size
    scaling_factor = board_diameter_pixels / 450.0  # 450mm is the actual diameter

    # Corrected dimensions scaled to the image size
    outer_circle_radius = int(scaling_factor)
    double_ring_outer_radius = int(170 * scaling_factor)
    double_ring_inner_radius = int(
        (170 - 10) * scaling_factor)  # 8mm ring width
    treble_ring_outer_radius = int(107 * scaling_factor)
    treble_ring_inner_radius = int(
        (107 - 10) * scaling_factor)  # 8mm ring width
    bullseye_outer_radius = int(31.8 * scaling_factor / 2)
    bullseye_inner_radius = int(12.7 * scaling_factor / 2)

    pts1 = np.float32(points)
    # Calculate the specific points
    pid = {
        "14/9": calculate_point(center, double_ring_outer_radius, 99),
        "8/16": calculate_point(center, double_ring_outer_radius, 189),
        "4/18": calculate_point(center, double_ring_outer_radius, 351),
        "6/10": calculate_point(center, double_ring_outer_radius, 45),
    }

    # Coordinates of the points in the target output
    # Adjust the size as needed
    pts2 = np.float32([[132, 263], [132, 537], [614, 187], [699, 448]])

    # The transformation matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (800, 800))

    # Draw circles
    cv2.circle(result, center, outer_circle_radius, (0, 255, 0), 1)
    cv2.circle(result, center, treble_ring_outer_radius, (0, 255, 0), 1)
    cv2.circle(result, center, treble_ring_inner_radius, (0, 255, 0), 1)
    cv2.circle(result, center, double_ring_outer_radius, (0, 255, 0), 1)
    cv2.circle(result, center, double_ring_inner_radius, (0, 255, 0), 1)
    cv2.circle(result, center, bullseye_outer_radius, (0, 255, 0), 1)
    cv2.circle(result, center, bullseye_inner_radius, (0, 255, 0), 1)

    # Draw radial lines
    # Dartboard has 20 segments, so 360/20 = 18 degrees per segment
    for angle in range(0, 360, 18):
        rad = math.radians((angle + 9) % 360)
        x_end = int(center[0] + double_ring_outer_radius * math.cos(rad))
        y_end = int(center[1] + double_ring_outer_radius * math.sin(rad))
        print(angle, rad, x_end, y_end)
        cv2.line(result, center, (x_end, y_end), (0, 255, 0), 1)

    # Mark the calculated points on the image
    for point in pid.values():
        cv2.circle(image, point, 3, (0, 0, 255), -1)
    # Display the result
    detection(matrix)


def calibrate():
    global img
    img = cv2.imread("./input/frame.png")
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detection(matrix):
    cap = cv2.VideoCapture("./input/2024-03-23 01-58-24.mkv")

    ret, background = cap.read()

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        fgmask = fgbg.apply(frame)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video Stream', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break


if __name__ == "__main__":
    calibrate()
