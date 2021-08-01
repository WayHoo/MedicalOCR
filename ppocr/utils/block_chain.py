# coding=utf-8
import math
from .block_process import get_box_key, get_center_point


def gen_block_chain(img, dt_boxes, rec_res):
    box_info_dict = {}
    for i in range(len(dt_boxes)):
        text, score = rec_res[i]
        box = dt_boxes[i]
        box_info = BoxInfo(box, text, score)
        box_info_dict[box_info.key] = box_info
    for key1, val1 in box_info_dict.items():
        for key2, val2 in box_info_dict.items():
            if key1 == key2:
                continue
            dis = get_box_dis(val1.box, val2.box)
            

def get_box_dis(box1, box2):
    p1 = get_center_point(box1)
    p2 = get_center_point(box2)
    return math.sqrt(math.pow(p2[0]-p1[0], 2) + math.pow(p2[1]-p1[1], 2))


class BoxInfo(object):

    def __init__(self, box, text, score):
        self.box = box
        self.text = text
        self.score = score
        self.key = get_box_key(box)
        self.left = {"key": "", "score": 0}
        self.right = {"key": "", "score": 0}
        self.up = {"key": "", "score": 0}
        self.down = {"key": "", "score": 0}

    def set_adjacent_box(self, dire, key, score):
        assert dire in {"left", "right", "up", "down"} and key != "" and score >= 0, "params error in set_adjacent_box"
        val = {"key": key, "score": score}
        if dire == "left":
            self.left = val
        elif dire == "right":
            self.right = val
        elif dire == "up":
            self.up = val
        elif dire == "down":
            self.down = val
