import cv2
import numpy as np
import math

calibration_points = []
calibration_done = False
ellipsis_vertices = []
new_points = []
intersectp_s = []
rotated_rect = []

cap = cv2.VideoCapture("./input/darts.mp4")
success, frame = cap.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

processing_frame = frame.copy()


def calibrate():
    global frame
    global calibration_done
    global processing_frame

    # Save calibration frame
    cv2.imwrite("./output/dartboard.png", frame)

    while not calibration_done:
        # Create new image for processing
        processing_frame = frame.copy()
        image_processing()

        calibration_done = True


def image_processing():
    print("Processing frame...")
    hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(hsv, -1, kernel)
    h, s, v = cv2.split(blur)

    cv2.imwrite("./output/hsv.png", hsv)
    cv2.imwrite("./output/blur.png", blur)
    cv2.imwrite("./output/h.png", h)
    cv2.imwrite("./output/s.png", s)
    cv2.imwrite("./output/v.png", v)

    # Thresholding
    thresh = cv2.threshold(v, 128, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite("./output/thresh.png", thresh)

    # Morphology -> Remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("./output/morph.png", morph)

    # Get only the edge
    edge = cv2.Canny(morph, 250, 255)
    cv2.imwrite("./output/edge.png", edge)

    # Find contours
    contours, hierarchy = cv2.findContours(morph, 1, 2)
    circle_radius = 0
    angle = 0
    center = (0, 0)
    axis = (0, 0)
    contours_done = False

    for cnt in contours:
        try:
            area = cv2.contourArea(cnt)
            if 200000/4 < area < 1000000/4:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(processing_frame, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]

                center = (int(x), int(y))
                axis = (int(a), int(b))

                a = int(a / 2)
                b = int(b / 2)

                circle_radius = a

                cv2.ellipse(processing_frame, center, (a, b),
                            angle, 0, 360, (255, 0, 0))

                xb = b * math.cos(angle)
                yb = b * math.sin(angle)

                xa = a * math.sin(angle)
                ya = a * math.cos(angle)

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(processing_frame, [box], 0, (0, 0, 255), 2)
                contours_done = True

        except ValueError as e:
            print("Error: ", e)

        if contours_done:
            cv2.imwrite("./output/contour.png", processing_frame)

            angle_zone_1 = (angle - 5, angle + 5)
            angle_zone_2 = (angle - 100, angle - 80)

            # Transform ellipse to circle
            height, width, _ = frame.shape

            angle = angle * math.pi / 180

            # Build transformation matrix
            # http://math.stackexchange.com/questions/619037/circle-affine-transformation
            R1 = np.array([[math.cos(angle), math.sin(angle), 0],
                           [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            R2 = np.array([[math.cos(angle), -math.sin(angle), 0],
                          [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

            T1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
            T2 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])

            D = np.array([[1, 0, 0], [0, axis[0] / axis[1], 0], [0, 0, 1]])

            M = T2.dot(R2.dot(D.dot(R1.dot(T1))))

            M_inv = np.linalg.inv(M)

            print(M)
            print(M_inv)

            # Fit line to find intersection point
            lines = cv2.HoughLines(edge, 1, np.pi / 70, 200)

            p = []
            lines_seg = []
            count = 0

            for rho, theta in lines[0]:
                print("rho, theta: ", rho, theta)
                print("angle_zone_1: ", angle_zone_1)
                print("np.pi /180 * angle_zone_1[0]: ",
                      np.pi / 180 * angle_zone_1[0])
                print(np.pi / 180 * angle_zone_1[0],
                      np.pi / 180 * angle_zone_1[1])
                if theta > np.pi / 180 * angle_zone_1[0] and theta < np.pi / 180 * angle_zone_1[1]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 3000 * (-b))
                    y1 = int(y0 + 3000 * (a))
                    x2 = int(x0 - 3000 * (-b))
                    y2 = int(y0 - 3000 * (a))

                    for rho1, theta1 in lines[0]:
                        if theta > np.pi / 180 * angle_zone_2[0] and theta < np.pi / 180 * angle_zone_2[1]:
                            a = np.cos(theta1)
                            b = np.sin(theta1)
                            x0 = a * rho1
                            y0 = b * rho1
                            x3 = int(x0 + 3000 * (-b))
                            y3 = int(y0 + 3000 * (a))
                            x4 = int(x0 - 3000 * (-b))
                            y4 = int(y0 - 3000 * (a))

                            if y1 == y2 and y3 == y4:
                                diff = abs(y1 - y3)
                            elif x1 == x2 and x3 == x4:
                                diff = abs(x1 - x3)
                            else:
                                diff = 0

                            if 0 < diff < 200:
                                continue

                            p.append((x1, y1))
                            p.append((x2, y2))
                            p.append((x3, y3))
                            p.append((x4, y4))

                            intersectpx, intersectpy = intersect_lines(
                                p[count], p[count + 1], p[count + 2], p[count + 3]
                            )

                            if (intersectpx < 100 or intersectpx > 800) or (intersectpy < 100 or intersectpy > 800):
                                continue

                            lines_seg.append([(x1, y1), (x2, y2)])
                            lines_seg.append([(x3, y3), (x4, y4)])

                            cv2.line(processing_frame, (x1, y1),
                                     (x2, y2), (255, 0, 0), 1)
                            cv2.line(processing_frame, (x3, y3),
                                     (x4, y4), (255, 0, 0), 1)

                            count += 4

            ellipsis_vertices.append(
                [(box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2])
            ellipsis_vertices.append(
                [(box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2])
            ellipsis_vertices.append(
                [(box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2])
            ellipsis_vertices.append(
                [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2])

            test_point_1 = M.dot(np.transpose(np.hstack([center, 1])))
            test_point_2 = M.dot(np.transpose(
                np.hstack([ellipsis_vertices[0], 1])))
            test_point_3 = M.dot(np.transpose(
                np.hstack([ellipsis_vertices[1], 1])))
            test_point_4 = M.dot(np.transpose(
                np.hstack([ellipsis_vertices[2], 1])))
            test_point_5 = M.dot(np.transpose(
                np.hstack([ellipsis_vertices[3], 1])))

            new_points.append([test_point_2[0], test_point_2[1]])
            new_points.append([test_point_3[0], test_point_3[1]])
            new_points.append([test_point_4[0], test_point_4[1]])
            new_points.append([test_point_5[0], test_point_5[1]])
            new_points.append([test_point_1[0], test_point_1[1]])

            lines_seg_done = False

            print("Lines segment: ", lines_seg)

            for line in lines_seg:
                line_p1 = M.dot(np.transpose(np.hstack([line[0], 1])))
                line_p2 = M.dot(np.transpose(np.hstack([line[1], 1])))
                inter1, inter_p1, inter2, inter_p2 = intersect_line_circle(
                    np.asarray(center), circle_radius, np.asarray(
                        line_p1), np.asarray(line_p2)
                )
                print("Intersection: ", inter1, inter_p1, inter2, inter_p2)
                if inter1:
                    inter_p1 = M_inv.dot(
                        np.transpose(np.hstack([inter_p1, 1])))
                    if inter2:
                        inter_p2 = M_inv.dot(
                            np.transpose(np.hstack([inter_p2, 1])))
                        intersectp_s.append(inter_p1)
                        intersectp_s.append(inter_p2)
                lines_seg_done = True

            print("Calibration points: ", calibration_points)
            print("Lines segment points: ", intersectp_s)

            if lines_seg_done:
                try:
                    # calculate mean val between: 0,4;1,5;2,6;3,7
                    new_intersect = np.mean(
                        ([intersectp_s[0], intersectp_s[4]]), axis=0, dtype=np.float32)
                    calibration_points.append(new_intersect)  # top
                    new_intersect = np.mean(
                        ([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
                    calibration_points.append(new_intersect)  # bottom
                    new_intersect = np.mean(
                        ([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
                    calibration_points.append(new_intersect)  # left
                    new_intersect = np.mean(
                        ([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
                    calibration_points.append(new_intersect)  # right
                except:
                    # take only first 4 arguments
                    pointarray = np.array(intersectp_s[:4])
                    top_idx = [np.argmin(pointarray[:, 1])][0]
                    pointarray_1 = np.delete(pointarray, [top_idx], axis=0)
                    bot_idx = [np.argmax(pointarray_1[:, 1])][0] + 1
                    pointarray_2 = np.delete(pointarray_1, [bot_idx], axis=0)
                    left_idx = [np.argmin(pointarray_2[:, 0])][0] + 2
                    right_idx = [np.argmax(pointarray_2[:, 0])][0] + 2

                    calibration_points.append(intersectp_s[top_idx])  # top
                    calibration_points.append(intersectp_s[bot_idx])  # bottom
                    calibration_points.append(intersectp_s[left_idx])  # left
                    calibration_points.append(intersectp_s[right_idx])  # right

                cv2.circle(processing_frame, (int(calibration_points[0][0]), int(
                    calibration_points[0][1])), 3, cv2.CV_RGB(255, 0, 0), 2, 8)
                cv2.circle(processing_frame, (int(calibration_points[1][0]), int(
                    calibration_points[1][1])), 3, cv2.CV_RGB(255, 0, 0), 2, 8)
                cv2.circle(processing_frame, (int(calibration_points[2][0]), int(
                    calibration_points[2][1])), 3, cv2.CV_RGB(255, 0, 0), 2, 8)
                cv2.circle(processing_frame, (int(calibration_points[3][0]), int(
                    calibration_points[3][1])), 3, cv2.CV_RGB(255, 0, 0), 2, 8)

                rotated_rect.append((box[1], box[2]))
                rotated_rect.append((box[2], box[3]))
                rotated_rect.append((box[0], box[3]))
                rotated_rect.append((box[0], box[1]))

                cv2.imwrite("./output/circles.png", processing_frame)


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

# Circle intersection


def intersect_line_circle(center, radius, p1, p2):
    baX = p2[0] - p1[0]
    baY = p2[1] - p1[1]
    caX = center[0] - p1[0]
    caY = center[1] - p1[1]

    a = baX * baX + baY * baY
    bBy2 = baX * caX + baY * caY
    c = caX * caX + caY * caY - radius * radius

    pBy2 = bBy2 / a
    q = c / a

    disc = pBy2 * pBy2 - q
    if disc < 0:
        return False, None, False, None

    tmpSqrt = math.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt

    pint1 = p1[0] - baX * abScalingFactor1, p1[1] - baY * abScalingFactor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - baX * abScalingFactor2, p1[1] - baY * abScalingFactor2
    return True, pint1, True, pint2


if __name__ == '__main__':
    print("Welcome to darts!")
    calibrate()
