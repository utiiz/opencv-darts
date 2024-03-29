import math


class Ellipse:
    def __init__(self):
        self.a = -1
        self.b = -1
        self.x = -1
        self.y = -1
        self.angle = -1


class CalibrationData:
    def __init__(self):
        # for perspective transform
        self.top = []
        self.bottom = []
        self.left = []
        self.right = []
        self.points = []
        # radii of the rings, there are 6 in total
        self.ring_radius = [14, 32, 194, 214, 320, 340]
        self.center_dartboard = (400, 400)
        self.sector_angle = 2 * math.pi / 20
        self.dst_points = []
        self.transformation_matrix = []
