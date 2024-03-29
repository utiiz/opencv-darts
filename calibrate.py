from video_capture import VideoStream
from classes import CalibrationData, Ellipse
import cv2
import numpy as np
import math

RIGHT = 0
LEFT = 1


def calibrate(cam_R):
    print("Calibrating...")
    success, frame = cam_R.read()

    image_calibration = frame.copy()
    image_original = frame.copy()

    cv2.imwrite("./output/cam_R.png", frame)

    calibration_done = False
    while not calibration_done:
        calibration_data = CalibrationData()
        calibration_data.points = get_transformation_points(
            image_calibration, RIGHT)

        calibration_done = True


def get_transformation_points(image, side):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(image_hsv, -1, kernel)
    h, s, v = cv2.split(blur)

    cv2.imwrite("./output/hsv.png", image_hsv)
    cv2.imwrite("./output/blur.png", blur)
    cv2.imwrite("./output/h.png", h)
    cv2.imwrite("./output/s.png", s)
    cv2.imwrite("./output/v.png", v)

    # Thresholding
    thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("./output/thresh.png", thresh)

    # Morphology -> Remove noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("./output/morph.png", morph)

    # Find ellipse
    ellipse, image = find_ellipse(morph, image)
    cv2.imwrite("./output/ellipse.png", image)

    # Canny
    edged = canny(morph)
    cv2.imwrite("./output/edged.png", edged)

    # Find 2 sectors lines
    if side == RIGHT:
        angle_zone_1 = (ellipse.angle - 5, ellipse.angle + 5)
        angle_zone_2 = (ellipse.angle - 100, ellipse.angle - 80)
        lines_seg, image = find_lines(
            edged, ellipse, image, angle_zone_1, angle_zone_2
        )
    else:
        lines_seg, image = find_lines(
            edged, ellipse, image, angle_zone_1=(80, 120), angle_zone_2=(30, 40)
        )

    cv2.imwrite("./output/lines.png", image)

    M = ellipse_to_circle(ellipse)
    intersectp_s = get_ellipse_line_intersection(ellipse, M, lines_seg)
    print("Intersection points: ", intersectp_s)

    source_points = []

    try:
        new_intersect = np.mean(
            ([intersectp_s[0], intersectp_s[4]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # top
        new_intersect = np.mean(
            ([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # bottom
        new_intersect = np.mean(
            ([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # left
        new_intersect = np.mean(
            ([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
        source_points.append(new_intersect)  # right
    except IndexError:
        pointarray = np.array(intersectp_s)
        top_idx = [np.argmin(pointarray[:, 1])][0]
        bot_idx = [np.argmax(pointarray[:, 1])][0]
        if side == RIGHT:
            left_idx = [np.argmin(pointarray[:, 0])][0]
            right_idx = [np.argmax(pointarray[:, 0])][0]
        else:
            left_idx = [np.argmax(pointarray[:, 0])][0]
            right_idx = [np.argmin(pointarray[:, 0])][0]
        source_points.append(intersectp_s[top_idx])  # top
        source_points.append(intersectp_s[bot_idx])  # bottom
        source_points.append(intersectp_s[left_idx])  # left
        source_points.append(intersectp_s[right_idx])  # right

    cv2.circle(image, (int(source_points[0][0]), int(
        source_points[0][1])), 3, (0, 0, 255), 2, 8)
    cv2.circle(image, (int(source_points[1][0]), int(
        source_points[1][1])), 3, (0, 0, 255), 2, 8)
    cv2.circle(image, (int(source_points[2][0]), int(
        source_points[2][1])), 3, (0, 0, 255), 2, 8)
    cv2.circle(image, (int(source_points[3][0]), int(
        source_points[3][1])), 3, (0, 0, 255), 2, 8)

    winName2 = "th circles?"
    cv2.namedWindow(winName2, cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow(winName2, image)

    end = cv2.waitKey(0)
    if end == 13:
        cv2.destroyAllWindows()
        return source_points


def find_ellipse(morph, image):
    ellipse = Ellipse()

    contours, hierarchy = cv2.findContours(
        morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours_done = False

    for contour in contours:
        try:
            area = cv2.contourArea(contour)
            if 200000 / 4 < area < 1000000 / 4:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]

                center = (int(x), int(y))

                a = a / 2
                b = b / 2

                cv2.ellipse(image, center, (int(a), int(b)),
                            angle, 0, 360, (0, 0, 255))

                contours_done = True
        except ValueError:
            print("Ellipse couldn't be found")

    if contours_done:
        ellipse.a = a
        ellipse.b = b
        ellipse.x = x
        ellipse.y = y
        ellipse.angle = angle

    return ellipse, image


def canny(image):
    return cv2.Canny(image, 250, 255)


def find_lines(edged, ellipse, image, angle_zone_1, angle_zone_2):
    p = []
    intersectp = []
    lines_seg = []
    counter = 0

    # fit line to find intersec point for dartboard center point
    lines = cv2.HoughLines(edged, 1, np.pi / 80, 100, 100)

    # Sector angles important -> make accessible
    for rho, theta in lines[0]:
        # split between horizontal and vertical lines (take only lines in certain range)
        if theta > np.pi / 180 * angle_zone_1[0] and theta < np.pi / 180 * angle_zone_1[1]:

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))

            for rho1, theta1 in lines[0]:

                if theta1 > np.pi / 180 * angle_zone_2[0] and theta1 < np.pi / 180 * angle_zone_2[1]:

                    a = np.cos(theta1)
                    b = np.sin(theta1)
                    x0 = a * rho1
                    y0 = b * rho1
                    x3 = int(x0 + 2000 * (-b))
                    y3 = int(y0 + 2000 * (a))
                    x4 = int(x0 - 2000 * (-b))
                    y4 = int(y0 - 2000 * (a))

                    if y1 == y2 and y3 == y4:  # Horizontal Lines
                        diff = abs(y1 - y3)
                    elif x1 == x2 and x3 == x4:  # Vertical Lines
                        diff = abs(x1 - x3)
                    else:
                        diff = 0

                    if diff < 200 and diff != 0:
                        continue

                    cv2.line(image, (x1, y1),
                             (x2, y2), (255, 0, 0), 1)
                    cv2.line(image, (x3, y3),
                             (x4, y4), (255, 0, 0), 1)

                    p.append((x1, y1))
                    p.append((x2, y2))
                    p.append((x3, y3))
                    p.append((x4, y4))

                    intersectpx, intersectpy = intersect_lines(
                        p[counter], p[counter + 1], p[counter + 2], p[counter + 3]
                    )

                    # consider only intersection close to the center of the image
                    if intersectpx < 200 or intersectpx > 900 or intersectpy < 200 or intersectpy > 900:
                        continue

                    intersectp.append((intersectpx, intersectpy))

                    lines_seg.append([(x1, y1), (x2, y2)])
                    lines_seg.append([(x3, y3), (x4, y4)])

                    cv2.line(image, (x1, y1),
                             (x2, y2), (255, 0, 0), 1)
                    cv2.line(image, (x3, y3),
                             (x4, y4), (255, 0, 0), 1)

                    # point offset
                    counter = counter + 4

    return lines_seg, image


# Line intersection
def intersect_lines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    x = (x1 + r * dx1 + x + s * dx) / 2.0
    y = (y1 + r * dy1 + y + s * dy) / 2.0
    return x, y


def ellipse_to_circle(ellipse):
    angle = (ellipse.angle) * math.pi / 180
    x = ellipse.x
    y = ellipse.y
    a = ellipse.a
    b = ellipse.b

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    R1 = np.array([[math.cos(angle), math.sin(angle), 0],
                  [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(angle), -math.sin(angle), 0],
                  [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

    return M


def get_ellipse_line_intersection(Ellipse, M, lines_seg):
    center_ellipse = (Ellipse.x, Ellipse.y)
    circle_radius = Ellipse.a
    M_inv = np.linalg.inv(M)

    # find line circle intersection and use inverse transformation matrix to transform it back to the ellipse
    intersectp_s = []
    for lin in lines_seg:
        line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
        line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
        inter1, inter_p1, inter2, inter_p2 = intersect_line_circle(np.asarray(center_ellipse), circle_radius,
                                                                   np.asarray(line_p1), np.asarray(line_p2))
        if inter1:
            inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            if inter2:
                inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
                intersectp_s.append(inter_p1)
                intersectp_s.append(inter_p2)

    return intersectp_s


def intersect_line_circle(center, radius, p1, p2):
    bX = p2[0] - p1[0]
    bY = p2[1] - p1[1]
    cX = center[0] - p1[0]
    cY = center[1] - p1[1]

    a = bX * bX + bY * bY
    bBy2 = bX * cX + bY * cY
    c = cX * cX + cY * cY - radius * radius

    pBy2 = bBy2 / a
    q = c / a

    disc = pBy2 * pBy2 - q
    if disc < 0:
        return False, None, False, None

    tmpSqrt = math.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt

    pint1 = p1[0] - bX * abScalingFactor1, p1[1] - bY * abScalingFactor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - bX * abScalingFactor2, p1[1] - bY * abScalingFactor2
    return True, pint1, True, pint2


if __name__ == '__main__':
    print("Welcome to darts!")

    cam_R = VideoStream(src="./input/darts.mp4").start()

    calibrate(cam_R)
