import cv2 as cv
import numpy as np
import math


class Ellipse:
    def __init__(self):
        self.x = 755
        self.y = 582
        self.w = 271
        self.h = 470
        self.angle = 154


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


window_name = "image"

image = cv.imread("./input/dartboard.png")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
kernel = np.ones((5, 5), np.float32) / 25
blur = cv.filter2D(hsv, -1, kernel)
h, s, v = cv.split(blur)
thresh = cv.threshold(v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
morph = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel)

ellipse = Ellipse()
line_1 = Line(665, 102, 747, 1001)
line_2 = Line(421, 476, 1106, 775)


def on_change_x(value):
    global ellipse
    ellipse.x = value
    on_change()


def on_change_y(value):
    global ellipse
    ellipse.y = value
    on_change()


def on_change_w(value):
    global ellipse
    ellipse.w = value
    on_change()


def on_change_h(value):
    global ellipse
    ellipse.h = value
    on_change()


def on_change_angle(value):
    global ellipse
    ellipse.angle = value
    on_change()


def on_change_line1_x1(value):
    global line_1
    line_1.x1 = value
    on_change()


def on_change_line1_y1(value):
    global line_1
    line_1.y1 = value
    on_change()


def on_change_line1_x2(value):
    global line_1
    line_1.x2 = value
    on_change()


def on_change_line1_y2(value):
    global line_1
    line_1.y2 = value
    on_change()


def on_change_line2_x1(value):
    global line_2
    line_2.x1 = value
    on_change()


def on_change_line2_y1(value):
    global line_2
    line_2.y1 = value
    on_change()


def on_change_line2_x2(value):
    global line_2
    line_2.x2 = value
    on_change()


def on_change_line2_y2(value):
    global line_2
    line_2.y2 = value
    on_change()


def on_change(intersecion_1=None, intersecion_2=None):
    try:
        # Create ellipse
        image = cv.imread("./input/dartboard.png")
        cv.ellipse(image, (ellipse.x, ellipse.y), (ellipse.w, ellipse.h),
                   ellipse.angle, 0, 360, (0, 255, 0), 1)

        # Create line
        cv.line(image, (line_1.x1, line_1.y1),
                (line_1.x2, line_1.y2), (255, 0, 0), 1)

        cv.line(image, (line_2.x1, line_2.y1),
                (line_2.x2, line_2.y2), (255, 255, 0), 1)

        if intersecion_1 is not None and intersecion_2 is not None:
            cv.circle(image, intersection1, 5, (0, 0, 255), -1)
            cv.circle(image, intersection2, 5, (0, 0, 255), -1)

        cv.imshow(window_name, image)
    except cv.error:
        pass


# Thresholding
on_change()
# Slider for ellipse
cv.createTrackbar("x", window_name, ellipse.x, 1920, on_change_x)
cv.createTrackbar("y", window_name, ellipse.y, 1920, on_change_y)
cv.createTrackbar("w", window_name, ellipse.w, 1920, on_change_w)
cv.createTrackbar("h", window_name, ellipse.h, 1920, on_change_h)
cv.createTrackbar("angle", window_name, ellipse.angle, 360, on_change_angle)

# Slider for line 1
cv.createTrackbar("line_1_x1", window_name,
                  line_1.x1, 1920, on_change_line1_x1)
cv.createTrackbar("line_1_y1", window_name,
                  line_1.y1, 1920, on_change_line1_y1)
cv.createTrackbar("line_1_x2", window_name,
                  line_1.x2, 1920, on_change_line1_x2)
cv.createTrackbar("line_1_y2", window_name,
                  line_1.y2, 1920, on_change_line1_y2)

# Slider for line 2
cv.createTrackbar("line_2_x1", window_name,
                  line_2.x1, 1920, on_change_line2_x1)
cv.createTrackbar("line_2_y1", window_name,
                  line_2.y1, 1920, on_change_line2_y1)
cv.createTrackbar("line_2_x2", window_name,
                  line_2.x2, 1920, on_change_line2_x2)
cv.createTrackbar("line_2_y2", window_name,
                  line_2.y2, 1920, on_change_line2_y2)

a = ellipse.w
b = ellipse.h
x0, y0 = ellipse.x, ellipse.y
x1, y1 = line_1.x1, line_1.y1
x2, y2 = line_1.x2, line_1.y2

# Ellipse equation: ((x - x0) / a)^2 + ((y - y0) / b)^2 = 1
# Line equation: y = m*x + c, where m = (y2 - y1) / (x2 - x1) and c = y1 - m*x1
m = (y2 - y1) / (x2 - x1)
c = y1 - m * x1

A = 1 / a**2 + m**2 / b**2
B = 2 * m * c / b**2 - 2 * x0 / a**2 - 2 * m * y0 / b**2
C = x0**2 / a**2 + c**2 / b**2 - 2 * c * y0 / b**2 + y0**2 / b**2 - 1

discriminant = B**2 - 4 * A * C

if discriminant >= 0:
    x_intercept1 = (-B + np.sqrt(discriminant)) / (2 * A)
    x_intercept2 = (-B - np.sqrt(discriminant)) / (2 * A)

    y_intercept1 = m * x_intercept1 + c
    y_intercept2 = m * x_intercept2 + c

    intersection1 = (int(x_intercept1), int(y_intercept1))
    intersection2 = (int(x_intercept2), int(y_intercept2))

    print("Intersection Point 1:", intersection1)
    print("Intersection Point 2:", intersection2)

    # Draw the intersection points on the image
    on_change(intersection1, intersection2)
else:
    print("No real intersection points")

cv.waitKey(0)
cv.destroyAllWindows()
