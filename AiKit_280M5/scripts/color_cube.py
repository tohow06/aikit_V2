import numpy as np

COLOR_LIST = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "cyan": 3,
    "yellow": 4
}

class ColorCube():
    def __init__(self,x,y,color):
        self.x = x
        self.y = y
        self.color = color
        self.color_id = COLOR_LIST[color]