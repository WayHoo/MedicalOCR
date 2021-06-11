# coding=utf-8
import math
import time
import cv2
import pandas as pd
import numpy as np
from scipy import optimize, stats
from utils.nlp import jieba_seg
from collections import deque
from PIL import Image, ImageDraw, ImageFont


def calc_block_angle(dt_boxes, rec_res):
    """
    根据文本检测框计算文本倾斜角度
    :param dt_boxes: 文本检测框列表
    [array([[262., 130.], [423., 130.], [423., 167.], [262., 167.]], dtype=float32), ...]
    :param rec_res: 文本块内容列表
    :return: 倾斜角度（弧度）
    """

    angles = []
    i = -1
    for box in dt_boxes:
        i += 1
        # 文本倾斜角
        angle = calc_box_angle(box)
        # 文本块宽高比
        ratio = calc_width_height_ratio(box)
        text, score = rec_res[i]
        if score >= 0.9 and 2.0 < ratio < 100.0:
            angles.append(angle)
        # print('(%.2f, %.2f)  (%.2f, %.2f)' % (x1, y1, x2, y2))
        print('angle=%.2f, ratio=%.2f, text=%s, score=%.2f, box=%s' % (angle, ratio, text, score, box))

    df = pd.DataFrame(angles, columns=['angle'])
    avg = df['angle'].mean()  # 计算均值
    # std = df['angle'].std()   # 计算标准差
    # _, p_value = stats.kstest(df['angle'], 'norm', (avg, std))
    print('avg_angle=%.2f' % avg)
    # print('p_value=%.2f' % p_value)
    return avg


def block_seg(img, dt_boxes):
    mask = np.zeros(img.shape, np.uint8) + 255
    boxes = []
    for box in dt_boxes:
        box = np.array(box, dtype=np.int32)
        boxes.append(box)
    cv2.fillPoly(mask, boxes, (0, 0, 0))
    return cv2.bitwise_or(img, mask)


def extract_test_sheet(img, dt_boxes, rec_res):
    """
    提取化验单内容
    :param img: 化验单图像
    :param dt_boxes: 检测文本框
    :param rec_res: 识别文本框
    :return: None
    """
    # 化验栏标题
    with open("./doc/dict/test_sheet_head_dict.txt", "r", encoding="utf-8") as f:
        head_words = f.read().splitlines()
    # 候选表头文本框，候选非表头文本框
    candi_head_box_dict, candi_other_box_dict, all_box_dict = {}, {}, {}
    head_box_dict = {}  # box_key → {"seg_num": 3, "words": ["结果", "提示", "范围"]}
    begin = time.time()
    print("jieba_seg begin...")
    for i in range(len(dt_boxes)):
        text, score = rec_res[i]
        box = dt_boxes[i]
        box_key = get_box_key(box)
        # ---------------------jieba_seg begin-----------------------
        words = jieba_seg(text)
        meta = {'text': text, 'score': score, 'box': box, "seg_words": words}
        all_box_dict[box_key] = meta
        for word in words:
            # 英文单词转换为大写
            upper_word = word.upper()
            if upper_word in head_words:
                if box_key not in head_box_dict.keys():
                    head_box_dict[box_key] = {"seg_num": len(words), "words": [word]}
                    candi_head_box_dict[box_key] = meta
                else:
                    head_box_dict[box_key]["words"].append(word)
                print("word=%s, text=%s, box=%s" % (word, text, box))
        if box_key not in candi_head_box_dict.keys():
            candi_other_box_dict[box_key] = meta
        # ---------------------jieba_seg end-----------------------
    end = time.time()
    print("jieba_seg end, cost %.2fs" % (end - begin))

    # TODO: 分词方式提取化验栏标题，可处理竖直方向多个化验表，水平方向不支持
    head_lines = extract_head_lines(candi_head_box_dict)
    line_f_1_list = []  # 表头直线方程列表
    table_cnt_dict = {}  # 记录化验表的栏数（单栏、双栏、三栏...）的dict
    for idx, line in enumerate(head_lines):
        k, b, has, exclude_boxes, sub_table_cnt = get_head_line_f_1(line, head_box_dict, img)
        if has:
            line_f_1_list.append((k, b))
            table_cnt_dict[get_line_key(k, b)] = sub_table_cnt
        for k, v in exclude_boxes.items():
            candi_other_box_dict[k] = v
    # 在候选非表头文本框中剔除竖向文本框
    vertical_box_dict, other_box_dict = {}, {}
    for box_key, meta in candi_other_box_dict.items():
        box = meta["box"]
        ratio = calc_width_height_ratio(box)
        angle = calc_box_angle(box)
        if ratio < 0.5 or angle > 45 or angle < -45:
            vertical_box_dict[box_key] = meta
        else:
            other_box_dict[box_key] = meta
    tables = classify_boxes(other_box_dict, line_f_1_list)
    for table in tables:
        k, b = table["f_1"]
        boxes = table["boxes"]
        sub_table_cnt = table_cnt_dict[get_line_key(k, b)]
        print("%d栏化验单" % sub_table_cnt)
        lines = split_horizon_lines(img, boxes, k, b)
    return


def get_middle_point(box):
    """
    获取四边形中心点坐标
    :param box: 文本框
    :return: 坐标
    """
    x, y = 0.0, 0.0
    for point in box:
        x += point[0]
        y += point[1]
    return x / 4.0, y / 4.0


def calc_point_to_line_dis(k, b, x, y):
    """
    计算点到直线的距离，直线方程 y = k*x + b
    :param k: 斜率
    :param b: 截距
    :param x: 自变量
    :param y: 因变量
    :return:
    """
    # 将斜截式转为一般式，a*x + b*y + c = 0
    a, b, c = k, -1, b
    return math.fabs(a * x + b * y + c) / math.sqrt(a * a + b * b)


def get_side_points(box):
    """
    获取四边形左侧边中点坐标和右侧边中点坐标
    :param box: 文本框
    :return: 两个点坐标
    """
    left_point = ((box[0][0] + box[3][0]) / 2.0, (box[0][1] + box[3][1]) / 2.0)
    right_point = ((box[1][0] + box[2][0]) / 2.0, (box[1][1] + box[2][1]) / 2.0)
    return left_point, right_point


def get_center_point(box):
    """
    计算矩形文本框的质心坐标
    :param box: 文本框
    :return: 文本框质心左边，横坐标为四个顶点横坐标的平均值，纵坐标为四个顶点纵坐标的平均值
    """
    (x1, y1), (x2, y2) = get_side_points(box)
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    return x, y


def f_1(x, k, b):
    """
    直线方程函数，y = k*x + b
    :param x: 自变量
    :param k: 斜率
    :param b: 截距
    :return: 因变量
    """
    return k * x + b


def get_box_width(box):
    """
    获取文本框宽度值
    :param box: 文本框
    :return: 宽度值
    """
    return (box[1][0] + box[2][0] - box[0][0] - box[3][0]) / 2.0


def get_box_height(box):
    """
    获取文本框高度值
    :param box: 文本框
    :return: 高度值
    """
    return (box[2][1] + box[3][1] - box[0][1] - box[1][1]) / 2.0


def calc_width_height_ratio(box):
    """
    计算文本框的高宽比
    :param box: 文本框
    :return: 高宽比
    """
    width = get_box_width(box)
    height = get_box_height(box)
    ratio = math.fabs(width / height)
    return ratio


def calc_box_angle(box):
    """
    计算文本框倾斜角
    :param box: 文本框
    :return: 角度值的倾斜角，一、四象限
    """
    (x1, y1), (x2, y2) = get_side_points(box)
    theta = math.atan2(y1 - y2, x2 - x1)
    # 将弧度制的角度换算到一、四象限
    if theta > math.pi / 2.0:
        theta -= math.pi
    elif theta < -math.pi / 2.0:
        theta += math.pi
    angle = math.degrees(theta)
    return angle


def get_point_project_to_line(point, a, b):
    """
    计算点投影到直线的点的坐标
    :param point: 点的 x, y 坐标，point为元组类型
    :param a: 直线方程参数（斜率），直线方程为 y = ax + b 形式
    :param b: 直线方程参数（截距）
    :return: 投影点整型坐标
    """
    x, y = point[0], point[1]
    x0 = (a*y + x - a*b) / (a*a + 1)
    y0 = a*x0 + b
    return int(x0), int(y0)


def get_box_line_dire(box, k, b):
    """
    判断直线与文本框方位。注意，坐标系建立在图像上，x正方向为 →，y正方向为 ↓
    :param box: 文本框
    :param k: 直线方程参数（斜率），直线方程为 y = k*x + b 形式
    :param b: 直线方程参数（截距）
    :return: 1-文本框在直线上方 0-直线穿过文本框 -1-文本框在直线下方
    """
    up, on, down = False, False, False
    for point in box:
        _x, _y = point[0], point[1]
        y = k*_x + b
        if _y > y:
            down = True
        elif _y == y:
            on = True
        else:
            up = True
    if on or (up and down):
        return 0
    elif up and not down:
        return 1
    else:
        return -1


def get_box_key(box):
    """
    获取文本框的 key，用于dict
    :param box: 文本框
    :return: 以1、4坐标点的x、y坐标拼接成的字符串
    """
    return "%d_%d_%d_%d" % (box[0][0], box[0][1], box[3][0], box[3][1])


def get_line_key(k, b):
    """
    获取直线的 key，用于dict
    :param k: 斜率
    :param b: 截距
    :return: 字符串
    """
    return "%.2f_%.2f" % (k, b)


def get_boxes_line_f_1(boxes, img=None, default_k=0):
    """
    计算多个文本框拟合的直线方程，宽（x方向）文本框取左右两边的中点，瘦文本框取质心
    :param boxes: 文本框列表
    :param img: 图片，传改参数即绘制直线
    :param default_k: 当文本框数量少于2个时，直线斜率使用默认斜率
    :return: 直线方程参数，k-斜率 b-截距
    """
    x_cors, y_cors = [], []
    draw_x1, draw_x2 = float("inf"), 0
    for box in boxes:
        ratio = calc_width_height_ratio(box)
        (x1, y1), (x2, y2) = get_side_points(box)
        if ratio >= 1:
            x_cors.extend([x1, x2])
            y_cors.extend([y1, y2])
        else:
            x_cors.append((x1 + x2) / 2.0)
            y_cors.append((y1 + y2) / 2.0)
        if x1 < draw_x1:
            draw_x1 = x1
        if x2 > draw_x2:
            draw_x2 = x2
    if len(x_cors) > 2:
        k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    else:
        avg_x, avg_y = np.mean(x_cors), np.mean(y_cors)
        k = default_k
        b = avg_y - k * avg_x
    if img is not None:
        draw_y1, draw_y2 = int(f_1(draw_x1, k, b)), int(f_1(draw_x2, k, b))
        cv2.line(img, (int(draw_x1), int(draw_y1)), (int(draw_x2), int(draw_y2)), (0, 0, 255), 2, cv2.LINE_AA)
    return k, b


def head_line_bfs(key="", box_dict=None):
    """
    bfs计算最长化验栏文本框列表
    :param key: box_key
    :param box_dict: {box_key: [box_key1, box_key2...], ...}
    :return: [box_key1, box_key2, ...]
    """
    if key not in box_dict.keys():
        return []
    res = [key]
    visited = {key}
    if len(box_dict[key]) == 0:
        return res
    queue = deque()
    for k in box_dict[key]:
        queue.append(k)
    while len(queue) > 0:
        k = queue.popleft()
        visited.add(k)
        res.append(k)
        if k in box_dict.keys() and len(box_dict[k]) > 0:
            for _k in box_dict[k]:
                if _k not in visited:
                    visited.add(_k)
                    queue.append(_k)
    return res


def extract_head_lines(candi_head_box_dict):
    """
    化验单表头文本框按行聚类
    :param candi_head_box_dict: 候选表头文本框，通过分词算法提取，{box_key: meta, ...}
    :return: 二维列表，[[meta1, meta2, ...], ...],
    meta元素类型为{"text": "项目单位", "score": 0.9, "box": [[1.0, 2.0], ...], "seg_words": ["项目", "单位"]}
    """
    same_line_box_dict = {}  # box→与box在同一行的box列表(不包含key对应的box)
    for box_key, meta in candi_head_box_dict.items():
        box = meta["box"]
        # 确定文本框的直线方程
        a, b = get_boxes_line_f_1([box])
        same_line_box_dict[box_key] = []
        for _box_key, _meta in candi_head_box_dict.items():
            _box = _meta["box"]
            if _box_key == box_key:
                continue
            if get_box_line_dire(_box, a, b) == 0:
                same_line_box_dict[box_key].append(_box_key)
    candi_lines = []
    for k, v in same_line_box_dict.items():
        meta = {"key": k, "box_keys": head_line_bfs(k, same_line_box_dict)}
        candi_lines.append(meta)
    candi_lines.sort(key=lambda x: len(x["box_keys"]), reverse=True)
    final_lines = []
    tmp_set = set()
    for meta in candi_lines:
        if meta["key"] in tmp_set:
            continue
        boxes = []
        for tmp_key in meta["box_keys"]:
            if tmp_key in tmp_set:
                continue
            else:
                tmp_set.add(tmp_key)
                boxes.append(candi_head_box_dict[tmp_key])
        if len(boxes) > 0:
            final_lines.append(boxes)
    return final_lines


def split_horizon_lines(img, boxes, k, b, threshold=0.6):
    """
    分割水平文本行
    :param img: 图像
    :param boxes: 表头直线下方的文本框列表，[meta1, meta2, ...]
    :param k: 表头直线斜率
    :param b: 表头直线截距
    :param threshold: 文本行分割阈值
    :return: [[meta1, meta2, ...], ...]
    """
    for box in boxes:
        x, y = get_center_point(box["box"])
        dis = calc_point_to_line_dis(k, b, x, y)
        box["dis"] = dis
    boxes = sorted(boxes, key=lambda t: t["dis"])
    line_idx = 0  # 行号
    pre_k = k  # 前一条直线斜率
    lines = []  # 每一行的文本框列表
    while True:
        if len(boxes) == 0:
            break
        line_idx += 1  # 行号
        box_num = 0  # 同一行文本框中的编号
        # height_sum: 同一行文本框的 y 方向高度累加值；用于计算 avg_height
        # avg_height: 判定为同一行的文本框的平均高度
        height_sum, avg_height = 0.0, 0.0
        pre_val = boxes[0]["dis"]  # 前一个文本框到直线的距离
        line_boxes = []  # 同一行的文本框列表
        for idx, box in enumerate(boxes):
            delta = box["dis"] - pre_val
            scale = 0
            # 使用当前文本框与前一个文本框到直线距离的差值 占 前面同一行文本框高度平均值的比例 来判断当前文本框是否属于当前行
            if avg_height != 0:
                scale = delta / avg_height
            # 边界 case，最后一个 box
            if scale < threshold and idx + 1 == len(boxes):
                line_boxes.append(box)
                scale = 1
            if scale >= threshold:
                # 对判定为同行的文本框从左到右排序
                line_boxes = sorted(line_boxes, key=lambda t: t["box"][0][0])
                lines.append(line_boxes)
                # 打印输出
                print("[LINE %d]............................" % line_idx)
                pure_boxes = []
                for tmp_box in line_boxes:
                    _box = tmp_box["box"]
                    pure_boxes.append(_box)
                    print("text=%s, dis=%.2f" % (tmp_box["text"], tmp_box["dis"]))
                # 当一行的文本框不少于2个时，才重新计算该行直线方程；否则沿用上一行的直线斜率
                _k, _b = get_boxes_line_f_1(pure_boxes, img, default_k=pre_k)
                pre_k = _k
                # 计算剩余文本框到该直线的距离
                boxes = boxes[len(line_boxes):]
                if len(boxes) == 0:
                    break
                for _box in boxes:
                    x, y = get_center_point(_box["box"])
                    dis = calc_point_to_line_dis(_k, _b, x, y)
                    _box["dis"] = dis
                boxes = sorted(boxes, key=lambda t: t["dis"])
                break
            else:
                # 没有换行，继续累加文本框
                box_num += 1
                line_boxes.append(box)
                height_sum += get_box_height(box["box"])
                avg_height = height_sum / box_num
                pre_val = box["dis"]
    return lines


def get_head_line_f_1(head_line, head_box_dict, img):
    """
    计算表头直线方程
    :param head_line: 二维列表，[[meta1, meta2, ...], ...],
    meta元素类型为{"text": "项目单位", "score": 0.9, "box": [[1.0, 2.0], ...], "seg_words": ["项目", "单位"]}
    :param head_box_dict: {box_key: {"seg_num": 3, "words": ["结果", "提示", "范围"]}, ...}
    :param img: 化验单图像
    :return: 直线斜率、直线截距、是否可构成直线、无法构成直线的文本框 dict、化验单子表数量
    """
    # k-斜率，b-截距，has_line-是否可构成直线，sub_table_cnt-化验单子表数量（单栏化验单/双栏化验单）
    k, b, has_line, sub_table_cnt = 0, 0, False, 0
    exclude_box_dict = {}  # {box_key: meta}, meta 同 head_line 中的 meta
    # 忽略仅有一个文本框的标题栏
    if len(head_line) == 1:
        box_key = get_box_key(head_line[0]["box"])
        exclude_box_dict[box_key] = head_line[0]
        print("head line ignored, text=%s" % head_line[0]["text"])
        return k, b, has_line, exclude_box_dict, sub_table_cnt
    print("head line start...")
    boxes, seg_ratio_sum, seg_text_set = [], 0.0, set()
    total_word_cnt, diff_word_cnt, word_set = 0.0, 0.0, set()
    for meta in head_line:
        box = meta["box"]
        box_key = get_box_key(box)
        seg_text_set.add(box_key)
        boxes.append(box)
        seg_num = head_box_dict[box_key]["seg_num"]
        total_word_cnt += seg_num
        for word in head_box_dict[box_key]["words"]:
            if word not in word_set:
                diff_word_cnt += 1
                word_set.add(word)
        seg_ratio_sum += (1.0 / seg_num)
        print("head_text=%s, head_words=%s" % (meta["text"], head_box_dict[box_key]["words"]))
    seg_ratio = seg_ratio_sum / len(seg_text_set)
    sub_table_ratio = total_word_cnt / diff_word_cnt
    # TODO: 化验单栏数判断阈值设定
    if sub_table_ratio - math.floor(sub_table_ratio) > 0.2:
        sub_table_cnt = int(math.ceil(sub_table_ratio))
    else:
        sub_table_cnt = int(math.floor(sub_table_ratio))
    if seg_ratio >= (2.0 / 3.0):
        k, b = get_boxes_line_f_1(boxes, img)
        has_line = True
        print("head line end, seg_ratio=%.2f" % seg_ratio)
    else:
        for meta in head_line:
            box_key = get_box_key(meta["box"])
            exclude_box_dict[box_key] = meta
        print("head line skipped, seg_ratio=%.2f" % seg_ratio)
    return k, b, has_line, exclude_box_dict, sub_table_cnt


def classify_boxes(box_dict, line_f_1_list):
    """
    根据表头直线在y方向的上下关系，对整个图像的文本框进行归类
    :param box_dict: {box_key: meta, ...}
    :param line_f_1_list: [(k1, b1), (k2, b2), ...]
    :return: [{"f_1": (k, b), "boxes": [meta1, meta2, ...]}, ...]
    """
    tables = []
    line_box_dire_dict = {}
    # 直线方程按截距从小到大排序（直线从高到低，x正方向为→，y正方向为↓）
    line_f_1_list = sorted(line_f_1_list, key=lambda t: t[1])
    for box_key, meta in box_dict.items():
        box = meta["box"]
        for k, b in line_f_1_list:
            line_key = get_line_key(k, b)
            dire = get_box_line_dire(box, k, b)
            if line_key not in line_box_dire_dict.keys():
                line_box_dire_dict[line_key] = {box_key: dire}
            else:
                line_box_dire_dict[line_key][box_key] = dire
    for idx, line in enumerate(line_f_1_list):
        meta_list = []
        line_key = get_line_key(line[0], line[1])
        below_lines = []
        if idx+1 < len(line_f_1_list):
            below_lines = line_f_1_list[idx+1:]
        for box_key, meta in box_dict.items():
            # 属于当前化验表的文本框的条件：在当前表头直线下方，并且在剩余直线上方
            if line_box_dire_dict[line_key][box_key] != -1:
                continue
            flag = True  # 当前文本框是否在剩余直线上方的标志，默认True
            for l in below_lines:
                tmp_line_key = get_line_key(l[0], l[1])
                if line_box_dire_dict[tmp_line_key][box_key] != 1:
                    flag = False
                    break
            if flag:
                meta_list.append(meta)
        table = {"f_1": line, "boxes": meta_list}
        tables.append(table)
    return tables
