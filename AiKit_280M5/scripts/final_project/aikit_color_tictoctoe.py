#!/usr/bin/env python3
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from color_cube import ColorCube
from pymycobot.mycobot import MyCobot
import platform

IS_CV_4 = cv2.__version__[0] == "4"
__version__ = "1.0"

class ObjectDetect:
    def __init__(self, camera_x=160, camera_y=15):
        # inherit the parent class
        super(ObjectDetect, self).__init__()

        # init camera
        if platform.system() == "Windows":
            self.cap_num = 0
            self.cap = cv2.VideoCapture(self.cap_num)
            if not self.cap.isOpened():
                self.cap.open(1)
        elif platform.system() == "Linux":
            self.cap_num = 4
            self.cap = cv2.VideoCapture(self.cap_num, cv2.CAP_V4L)
            if not self.cap.isOpened():
                self.cap.open()
        self.init_num = 0
        self.nparams = 0

        # declare mycobot280
        self.mc = None
        # get real serial
        self.plist = [
            str(x).split(" - ")[0].strip() for x in serial.tools.list_ports.comports()
        ]
        # home position
        self.home_angles = [0.61, 45.87, -92.37, -41.3, 2.02, 9.58]
        self.home_coords = [0, 0, 0, 0, 0, 0]
        self.standby_coords = [0, 0, 0, 0, 0, 0]

        # 移动角度
        self.move_angles = [
            [0.61, 45.87, -92.37, -41.3, 2.02, 9.58],  # init the point
            [18.8, -7.91, -54.49, -23.02, -0.79, -14.76],  # point to grab
        ]

        # 移动坐标
        self.move_coords = {
            # D Sorting area
            "D": [132.2, -136.9, 200.8, -178.24, -3.72, -107.17],
            "C": [238.8, -124.1, 204.3, -169.69, -5.52, -96.52],  # C Sorting area
            "A": [115.8, 177.3, 210.6, 178.06, -0.92, -6.11],  # A Sorting area
            "B": [-6.9, 173.2, 201.5, 179.93, 0.63, 33.83],  # B Sorting area
        }

        # set color HSV
        self.HSV = OrderedDict(
            {
                "red": [np.array([0, 43, 46]), np.array([8, 255, 255])],
                "green": [np.array([35, 43, 35]), np.array([90, 255, 255])],
                "blue": [np.array([100, 43, 46]), np.array([124, 255, 255])],
                "cyan": [np.array([78, 43, 46]), np.array([99, 255, 255])],
                "yellow": [np.array([11, 85, 70]), np.array([59, 255, 245])],
                # "yellow": [np.array([22, 93, 0]), np.array([45, 255, 245])],
            }
        )

        self.color_cubes = []
        self.color_cubes_count = np.zeros(len(self.HSV))
        self.real_cubes = []

        # choose place to set cube 选择放置立方体的地方
        self.color = 0
        # parameters to calculate camera clipping parameters 计算相机裁剪参数的参数
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        self.border_l = self.border_r = self.border_up = self.border_down = 0
        # set cache of real coord 设置真实坐标的缓存
        self.cache_x = self.cache_y = 0
        self.cache_cubes = []
        for c in self.HSV.keys():
            self.cache_cubes.append(ColorCube(0, 0, c))

        # use to calculate coord between cube and mycobot280
        # 用于计算立方体和 mycobot 之间的坐标
        self.sum_x1 = self.sum_x2 = self.sum_y2 = self.sum_y1 = 0
        # The coordinates of the grab center point relative to the mycobot280
        # 抓取中心点相对于 mycobot 的坐标
        self.camera_x, self.camera_y = camera_x, camera_y
        # The coordinates of the cube relative to the mycobot280
        # 立方体相对于 mycobot 的坐标
        self.c_x, self.c_y = 0, 0
        # The ratio of pixels to actual values
        # 像素与实际值的比值
        self.ratio = 0

        # Get ArUco marker dict that can be detected.
        # 获取可以检测到的 ArUco 标记字典。
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Get ArUco marker params. 获取 ArUco 标记参数
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        """
        # tic toc toe board
        """
        self.hborder = []
        self.vborder = []
        self.interest_points_xy = []
        self.interest_points_real_xy = []

    def calibrate(self):
        while cv2.waitKey(1) < 0:
            # read camera
            _, frame = self.cap.read()
            # deal img
            frame = self.transform_frame(frame)

            # calculate the parameters of camera clipping 计算相机裁剪的参数
            if self.init_num < 20:
                if self.get_calculate_params(frame, 'crop') is None:
                    cv2.imshow("calibrate", frame)
                    print("can't find two aruco")
                    continue
                else:
                    x1, x2, y1, y2 = self.get_calculate_params(frame, 'crop')
                    # self.draw_marker(frame, x1, y1)
                    # self.draw_marker(frame, x2, y2)
                    self.sum_x1 += x1
                    self.sum_x2 += x2
                    self.sum_y1 += y1
                    self.sum_y2 += y2
                    self.init_num += 1
                    continue
            elif self.init_num == 20:
                self.set_cut_params(
                    (self.sum_x1) / 20.0,
                    (self.sum_y1) / 20.0,
                    (self.sum_x2) / 20.0,
                    (self.sum_y2) / 20.0,
                )
                self.sum_x1 = self.sum_x2 = self.sum_y1 = self.sum_y2 = 0
                self.init_num += 1
                continue

            # calculate params of the coords between cube and mycobot280 计算立方体和 mycobot 之间坐标的参数
            if self.nparams < 10:
                if self.get_calculate_params(frame, 'center') is None:
                    cv2.imshow("calibrate", frame)
                    print("can't find two aruco after crop")
                    continue
                else:
                    x1, x2, y1, y2 = self.get_calculate_params(frame, 'center')
                    self.draw_marker(frame, x1, y1)
                    self.draw_marker(frame, x2, y2)
                    self.sum_x1 += x1
                    self.sum_x2 += x2
                    self.sum_y1 += y1
                    self.sum_y2 += y2
                    self.nparams += 1
                    continue
            elif self.nparams == 10:
                self.nparams += 1
                # calculate and set params of calculating real coord between cube and mycobot280
                # 计算和设置计算立方体和mycobot之间真实坐标的参数
                self.set_params(
                    (self.sum_x1 + self.sum_x2) / 20.0,
                    (self.sum_y1 + self.sum_y2) / 20.0,
                    abs(self.sum_x1 - self.sum_x2) / 10.0
                    + abs(self.sum_y1 - self.sum_y2) / 10.0,
                )
                print("complete params calculation")
                cv2.destroyAllWindows()
                break
                # continue

            cv2.imshow("calibrate", frame)

            # close the window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.cap.release()
                cv2.destroyAllWindows()
                sys.exit()
        pass

    def coords_above_xy(self, x, y):
        return [x, y, 170.6, 179.87, -3.78, -62.75]

    def coords_at_xy(self, x, y):
        return [x, y, 103, 179.87, -3.78, -62.75]

    def get_ab_cube_coords(self, cube):
        return [cube.x, cube.y, 170.6, 179.87, -3.78, -62.75]

    def get_at_cube_coords(self, cube):
        return [cube.x, cube.y, 103, 179.87, -3.78, -62.75]

    # 开启吸泵 m5
    def pump_on(self):
        # 让2号位工作
        self.mc.set_basic_output(2, 0)
        # 让5号位工作
        self.mc.set_basic_output(5, 0)

    # 停止吸泵 m5
    def pump_off(self):
        # 让2号位停止工作
        self.mc.set_basic_output(2, 1)
        # 让5号位停止工作
        self.mc.set_basic_output(5, 1)

    def place_tictoctoe(self, grid_num):
        """
        Pick up the cube from selected bins and place it on the board grid according to the grid number.
        """
        pick_and_standby_path = [self.move_coords["A"], self.standby_coords]
        for i in range(len(pick_and_standby_path)):
            self.mc.send_coords(pick_and_standby_path[i], 25, 1)
            if i == 2:
                self.pump_on()
                time.sleep(1.5)
            time.sleep(3)

        x, y = self.interest_points_real_xy[grid_num - 1]
        abCoords = self.coords_above_xy(x, y)
        atCoords = self.coords_at_xy(x, y)
        place_and_standby_path = [
            abCoords, atCoords, abCoords, self.standby_coords]
        for i in range(len(place_and_standby_path)):
            self.mc.send_coords(place_and_standby_path[i], 25, 1)
            if i == 1:
                self.pump_off()
                time.sleep(1.5)
            time.sleep(3)

    def move(self):
        top_cube = self.real_cubes[1]
        base_cube = self.real_cubes[0]
        print(
            f"Start stacking {top_cube.color} cube on {base_cube.color} cube")
        # 到抓取預備位置
        self.mc.send_angles(self.move_angles[1], 25)
        time.sleep(3)

        # send coordinates to move mycobot
        # 到木塊上方
        ab_cube = self.get_ab_cube_coords(top_cube)
        self.mc.send_coords(ab_cube, 25, 1)
        time.sleep(3)

        # 接觸木塊
        at_cube = self.get_at_cube_coords(top_cube)
        self.mc.send_coords(at_cube, 25, 0)
        time.sleep(3)

        # open pump
        self.pump_on()
        time.sleep(1.5)

        tmp = []
        while True:
            if not tmp:
                tmp = self.mc.get_angles()
            else:
                break
        time.sleep(0.5)

        # 抬起木塊
        self.mc.send_angles([tmp[0], -0.71, -54.49, -23.02, -0.79, tmp[5]], 25)
        time.sleep(3)

        # 移動到base_cube上方
        ab_base_cube = self.get_ab_cube_coords(base_cube)
        self.mc.send_coords(ab_base_cube, 25, 1)
        time.sleep(3)

        # 下降到top_cube接觸base_cube
        at_base_cube = self.get_at_cube_coords(base_cube)
        at_base_cube[2] += 70
        self.mc.send_coords(at_base_cube, 25, 1)
        time.sleep(3)

        # close pump
        self.pump_off()
        time.sleep(5)

        self.mc.send_angles(self.move_angles[0], 25)
        time.sleep(4.5)
        pass

    def decide_move(self):
        """
        Detect the cube status move or run 检测立方体状态移动或运行，直到立方體不再移動才開始抓取
        """
        moving = 0
        for rc in self.real_cubes:
            cache_cube = self.cache_cubes[rc.color_id]
            x = rc.x
            y = rc.y
            cache_x = cache_cube.x
            cache_y = cache_cube.y
            if (abs(x - cache_x) + abs(y - cache_y)) / 2 > 5:  # mm
                self.cache_cubes[rc.color_id] = rc
                print(f"{rc.color} cube is moving !!")
                moving = 1

        if moving:
            self.real_cubes = []
            return
        else:
            for c in self.HSV.keys():
                self.cache_cubes.append(ColorCube(0, 0, c))
            self.move()

    def run(self):
        """
        Init mycobot280 to home position
        """
        self.mc = MyCobot(self.plist[0], 115200)
        self.pump_off()
        time.sleep(1.5)
        self.mc.send_angles(self.home_angles, 20)
        time.sleep(2.5)

    def draw_marker(self, img, x, y):
        """
        - draw aruco
        """
        # draw rectangle on img 在 img 上绘制矩形
        cv2.rectangle(
            img,
            (x - 20, y - 20),
            (x + 20, y + 20),
            (0, 255, 0),
            thickness=2,
            lineType=cv2.FONT_HERSHEY_COMPLEX,
        )
        # add text on rectangle
        cv2.putText(
            img,
            "({},{})".format(x, y),
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (243, 0, 0),
            2,
        )

    def get_calculate_params(self, img, mode):
        """
        get points of two arucos\n
        获得两个 aruco 的点位
        """
        # Convert the image to a gray image 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect ArUco marker.
        corners, ids, rejectImaPoint = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        """
        Two Arucos must be present in the picture and in the same order.
        There are two Arucos in the Corners, and each aruco contains the pixels of its four corners.
        Determine the center of the aruco by the four corners of the aruco.
        """
        if len(corners) > 0:
            if ids is not None:
                if len(corners) <= 1 or ids[0] == 1:
                    return None
                x1 = x2 = y1 = y2 = 0
                point_11, point_21, point_31, point_41 = corners[0][0]
                point_1, point_2, point_3, point_4 = corners[1][0]

                if mode == 'crop':
                    x1 = np.min(np.array([point_11[0], point_21[0], point_31[0], point_41[0],
                                          point_1[0], point_2[0], point_3[0], point_4[0]]))
                    x2 = np.max(np.array([point_11[0], point_21[0], point_31[0], point_41[0],
                                          point_1[0], point_2[0], point_3[0], point_4[0]]))
                    y2 = np.min(np.array([point_11[1], point_21[1], point_31[1], point_41[1],
                                          point_1[1], point_2[1], point_3[1], point_4[1]]))
                    y1 = np.max(np.array([point_11[1], point_21[1], point_31[1], point_41[1],
                                          point_1[1], point_2[1], point_3[1], point_4[1]]))
                elif mode == 'center':
                    x1, y1 = int((point_11[0] + point_21[0] + point_31[0] + point_41[0]) / 4.0), int(
                        (point_11[1] + point_21[1] + point_31[1] + point_41[1]) / 4.0)
                    x2, y2 = int((point_1[0] + point_2[0] + point_3[0] + point_4[0]) / 4.0), int(
                        (point_1[1] + point_2[1] + point_3[1] + point_4[1]) / 4.0)

                return x1, x2, y1, y2
        return None

    def set_cut_params(self, x1, y1, x2, y2):
        """
        set camera clipping parameters 设置相机裁剪参数
        """
        self.border_l = int(x1)
        self.border_down = int(y1)
        self.border_r = int(x2)
        self.border_up = int(y2)

    def set_params(self, c_x, c_y, ratio):
        """
        - set parameters to calculate the coords between cube and mycobot280
        - 设置参数以计算立方体和 mycobot 之间的坐标
        """
        self.c_x = c_x
        self.c_y = c_y
        self.ratio = 220.0 / ratio

    def get_position(self, x, y):
        """
        calculate the coords between cube and mycobot280 计算立方体和 mycobot 之间的坐标
        """
        return ((y - self.c_y) * self.ratio + self.camera_x), (
            (x - self.c_x) * self.ratio + self.camera_y
        )

    def get_real_cube(self):
        """
        Calculate the mean of x and y in self.color_cubes with the same color
        and then calculate the coords between cube and mycobot280 save in the self.real_cube
        """
        index = self.color_cubes_count > 20
        unique_colors = np.array(list(self.HSV.keys()))[index]

        for color in unique_colors:
            sameColorCubes = [
                cube for cube in self.color_cubes if cube.color == color]
            total_x = sum(cube.x for cube in sameColorCubes)
            average_x = total_x / len(sameColorCubes)

            total_y = sum(cube.y for cube in sameColorCubes)
            average_y = total_y / len(sameColorCubes)

            position = self.get_position(average_x, average_y)
            new_cube = ColorCube(position[0], position[1], color)
            self.real_cubes.append(new_cube)

        self.real_cubes = sorted(
            self.real_cubes, key=lambda cube: cube.color_id)
        self.color_cubes = []
        self.color_cubes_count = np.zeros(len(self.HSV))

    def detect_board(self, c_color='red', h_color='blue'):
        """
        Detect the board and return the board matrix.

        Args:
            c_color: color of computer
            h_color: color of human
        """

        board = np.array([0 for _ in range(9)])

        # calculate board grid border line
        w = int((self.border_r - self.border_l))
        h = int((self.border_down - self.border_up))
        self.vborder.append([(w//3, 0), (w//3, h)])
        self.vborder.append([(w//3*2, 0), (w//3*2, h)])
        self.hborder.append([(0, h//3), (w, h//3)])
        self.hborder.append([(0, h//3*2), (w, h//3*2)])
        # calculate interest points
        step_w = w//3//2
        step_h = h//3//2
        self.interest_points_xy = [
            (step_w, step_h), (step_w*3, step_h), (step_w*5, step_h),
            (step_w, step_h*3), (step_w*3, step_h*3), (step_w*5, step_h*3),
            (step_w, step_h*5), (step_w*3, step_h*5), (step_w*5, step_h*5)
        ]
        for x, y in self.interest_points_xy:
            self.interest_points_real_xy.append(self.get_position(x, y))

        _, frame = self.cap.read()
        crop = self.transform_frame(frame)

        # draw board grid
        cv2.line(crop, self.vborder[0][0], self.vborder[0][1], (255, 0, 0), 1)
        cv2.line(crop, self.vborder[1][0], self.vborder[1][1], (255, 0, 0), 1)
        cv2.line(crop, self.hborder[0][0], self.hborder[0][1], (255, 0, 0), 1)
        cv2.line(crop, self.hborder[1][0], self.hborder[1][1], (255, 0, 0), 1)

        # detect interested points' color
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        for i in range(9):
            for mycolor, item in self.HSV.items():
                # if detected, set board matrix by c_color and h_color
                if np.all(hsv[self.interest_points_xy[i]] >= item[0]) and np.all(hsv[self.interest_points_xy[i]] <= item[1]):
                    if mycolor == c_color:
                        board[i] = -1
                    elif mycolor == h_color:
                        board[i] = 1
                    break
                else:
                    continue

        while cv2.waitKey(1) < 0:
            cv2.imshow('detect board', crop)

        return board.reshape(3, 3)

        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         self.cap.release()
        #         cv2.destroyAllWindows()
        #         sys.exit()

        # return board.reshape(3,3)

    def transform_frame(self, frame):
        """
        Calibrate the camera according to the calibration parameters.

        Enlarge the video pixel by 1.5 times, which means enlarge the video size by 1.5 times.

        If two ARuco values have been calculated, clip the video.
        """
        # enlarge the image by 1.5 times
        fx = 1.5
        fy = 1.5
        frame = cv2.resize(frame, (0, 0), fx=fx, fy=fy,
                           interpolation=cv2.INTER_CUBIC)
        # if self.x1 != self.x2:
        # the cutting ratio here is adjusted according to the actual situation
        # print("clip the video along the ARuco")
        # frame = frame[
        #     int(self.y2 * 0.78) : int(self.y1 * 1.1),
        #     int(self.x1 * 0.86) : int(self.x2 * 1.08),
        # ]
        if self.border_up != self.border_down:
            frame = frame[
                int(self.border_up): int(self.border_down),
                int(self.border_l): int(self.border_r),
            ]
        return frame

    def color_detect(self, img):
        """Detect cube color"""
        # set the arrangement of color'HSV
        x = y = 0
        raw = img.copy()
        for mycolor, item in self.HSV.items():
            # print("mycolor:",mycolor)
            redLower = np.array(item[0])
            redUpper = np.array(item[1])

            # transfrom the img to model of hsv 将图像转换为hsv模型
            hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

            # wipe off all color expect color in range 擦掉所有颜色期望范围内的颜色
            mask = cv2.inRange(hsv, item[0], item[1])

            # a etching operation on a picture to remove edge roughness
            # 对图片进行蚀刻操作以去除边缘粗糙度
            erosion = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=2)

            # the image for expansion operation, its role is to deepen the color depth in the picture
            # 用于扩展操作的图像，其作用是加深图片中的颜色深度
            dilation = cv2.dilate(erosion, np.ones(
                (1, 1), np.uint8), iterations=2)

            # adds pixels to the image 向图像添加像素
            # target = cv2.bitwise_and(img, img, mask=dilation)

            # the filtered image is transformed into a binary image and placed in binary
            # 将过滤后的图像转换为二值图像并放入二值
            # ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)

            # get the contour coordinates of the image, where contours is the coordinate value, here only the contour is detected
            # 获取图像的轮廓坐标，其中contours为坐标值，这里只检测轮廓
            contours, hierarchy = cv2.findContours(
                dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:
                # do something about misidentification
                boxes = [
                    box
                    for box in [cv2.boundingRect(c) for c in contours]
                    if min(img.shape[0], img.shape[1]) / 10
                    < min(box[2], box[3])
                    < min(img.shape[0], img.shape[1]) / 1
                ]
                if boxes:
                    for box in boxes:
                        x, y, w, h = box
                    # find the largest object that fits the requirements 找到符合要求的最大对象
                    c = max(contours, key=cv2.contourArea)
                    # get the lower left and upper right points of the positioning object
                    # 获取定位对象的左下和右上点
                    x, y, w, h = cv2.boundingRect(c)
                    # locate the target by drawing rectangle 通过绘制矩形来定位目标
                    img = cv2.rectangle(
                        img, (x, y), (x + w, y + h), (153, 153, 0), 2)
                    # calculate the rectangle center 计算矩形中心
                    x, y = (x * 2 + w) / 2, (y * 2 + h) / 2
                    # calculate the real coordinates of mycobot280 relative to the target
                    #  计算 mycobot 相对于目标的真实坐标

                    # 判断是否正常识别
                    if abs(x) + abs(y) <= 0:
                        continue
                    else:
                        self.color_cubes.append(ColorCube(x, y, mycolor))
                        color_index = list(self.HSV.keys()).index(mycolor)
                        self.color_cubes_count[color_index] += 1


if __name__ == "__main__":
    detection = ObjectDetect()
    detection.calibrate()
    input()
    b = detection.detect_board()
    print(b)
