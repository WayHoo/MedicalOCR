# coding=utf-8
import Levenshtein
import time


if __name__ == "__main__":
    source = "内氨酸基传移酶e"
    target = "丙氨酸氨基转移酶E"
    dist = Levenshtein.distance(source, target)
    print("dist=%d" % dist)
    ops = Levenshtein.editops(source, target)
    print("ops=%s" % ops)
