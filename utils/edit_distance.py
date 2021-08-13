# coding=utf-8
import Levenshtein
import time


def calc_edit_dist(source, target):
    return Levenshtein.distance(source, target)


if __name__ == "__main__":
    source = "内氨酸氨基转移酶"
    target = "丙氨酸氨基转移酶"
    dist = calc_edit_dist(source, target)
    print("dist=%d" % dist)
