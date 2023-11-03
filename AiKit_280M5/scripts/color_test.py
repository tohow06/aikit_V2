import sys

import cv2
import numpy as np
from aikit_color_stack import ObjectDetect

if __name__ == "__main__":
    import platform

    # open the camera
    if platform.system() == "Windows":
        cap_num = 0
        cap = cv2.VideoCapture(cap_num)
        if not cap.isOpened():
            cap.open(1)
    elif platform.system() == "Linux":
        cap_num = 4
        cap = cv2.VideoCapture(cap_num, cv2.CAP_V4L)
        if not cap.isOpened():
            cap.open()

    # init a class of ObjectDetect
    detect = ObjectDetect()

    _init_ = 20
    init_num = 0
    nparams = 0
    num = 0
    real_sx = real_sy = 0
    while cv2.waitKey(1) < 0:
        # read camera
        _, frame = cap.read()
        # deal img
        frame = detect.transform_frame(frame)
        if _init_ > 0:
            _init_ -= 1
            continue

        # get detect result 获取检测结果
        detect.color_detect(frame)
        print(detect.color_cubes_count)
        # 有兩種顏色以上的方塊被辨識到20次，才使用平均後的方塊當作real_cube
        if np.sum(detect.color_cubes_count > 20) >= 2:
            print("find two kinds of cubes")
            detect.get_real_cube()
            detect.decide_move()
            num = real_sx = real_sy = 0
        else:
            pass

        cv2.imshow("figure", frame)

        # close the window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
