# coding=utf-8
import math
import time
import cv2
import pandas as pd
import numpy as np
from scipy import optimize, stats
from utils.nlp import jieba_seg
from utils.u_str import calc_valid_char, str_len
from utils.sheet_process import parse_sheet_to_excel
from utils.head_cfg import is_cfg_head_word, is_cfg_key_word, get_category
from kie.corrector.correct import single_correct, multi_correct
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
    angles, i = [], -1
    for box in dt_boxes:
        i += 1
        angle = calc_box_angle(box)
        ratio = calc_aspect_ratio(box)
        text, score = rec_res[i]
        if score >= 0.9 and 2.0 < ratio < 100.0:
            angles.append(angle)
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


def extract_test_sheet(img, args, file_name, dt_boxes, rec_res):
    """
    提取化验单内容
    :param img: 化验单图像
    :param args: 配置参数
    :param file_name: 化验单文件名称，不包含后缀，例如 test_sheet (1)
    :param dt_boxes: 检测文本框
    :param rec_res: 识别文本框
    :return: None
    """
    # 候选表头文本框，候选非表头文本框
    candi_head_box_dict, candi_other_box_dict, all_box_dict = {}, {}, {}  # box_key → {meta}
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
            if is_cfg_head_word(upper_word):
                if box_key not in head_box_dict.keys():
                    head_box_dict[box_key] = {"seg_num": len(words), "words": [upper_word]}
                    candi_head_box_dict[box_key] = meta
                else:
                    head_box_dict[box_key]["words"].append(upper_word)
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
    head_line_dict = {}  # 表头直线方程 -> 表头文本框列表
    for idx, line in enumerate(head_lines):
        k, b, has, exclude_boxes, sub_table_cnt = get_head_line_f_1(line, head_box_dict, img)
        if has:
            line_f_1_list.append((k, b))
            line_key = get_line_key(k, b)
            head_line_dict[line_key] = line
            table_cnt_dict[line_key] = sub_table_cnt
        for k, v in exclude_boxes.items():
            candi_other_box_dict[k] = v
    # 在候选非表头文本框中剔除竖向文本框
    vertical_box_dict, other_box_dict = {}, {}
    for box_key, meta in candi_other_box_dict.items():
        box = meta["box"]
        ratio = calc_aspect_ratio(box)
        angle = calc_box_angle(box)
        if ratio < 0.5 or angle > 45 or angle < -45:
            vertical_box_dict[box_key] = meta
        else:
            other_box_dict[box_key] = meta
    tables = classify_boxes(other_box_dict, line_f_1_list)
    for i, table in enumerate(tables):
        k, b = table["f_1"]
        boxes = table["boxes"]
        line_key = get_line_key(k, b)
        sub_table_cnt = table_cnt_dict[line_key]
        print("%d栏化验单" % sub_table_cnt)
        lines = split_horizon_lines(img, boxes, k, b)
        csv_head_words = split_vertical_lines(lines, head_line_dict[line_key], head_box_dict)
        for idx, line in enumerate(lines):
            print("[LINE %d]............................" % idx)
            for meta in line:
                head_attrs = meta["attrs"] if "attrs" in meta else []
                print("text=%s, seg_words=%s, attrs=%s, score=%.2f" % (meta["text"], meta["seg_words"], head_attrs, meta["score"]))
        sheet_name = "化验单"+str(i+1) if len(tables) > 1 else "化验单"
        parse_sheet_to_excel(csv_head_words, lines, file_name, args.save_path, sheet_name)
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
    return x/4.0, y/4.0


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
    return math.fabs(a*x + b*y + c) / math.sqrt(a*a + b*b)


def get_side_points(box):
    """
    获取四边形左侧边中点坐标和右侧边中点坐标
    :param box: 文本框
    :return: 两个点坐标
    """
    left_point = [(box[0][0] + box[3][0]) / 2.0, (box[0][1] + box[3][1]) / 2.0]
    right_point = [(box[1][0] + box[2][0]) / 2.0, (box[1][1] + box[2][1]) / 2.0]
    return left_point, right_point


def get_center_point(box):
    """
    计算矩形文本框的质心坐标
    :param box: 文本框
    :return: 文本框质心左边，横坐标为四个顶点横坐标的平均值，纵坐标为四个顶点纵坐标的平均值
    """
    [x1, y1], [x2, y2] = get_side_points(box)
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
    return k*x + b


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


def calc_aspect_ratio(box):
    """
    计算文本框的横纵比（高宽比）
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
    [x1, y1], [x2, y2] = get_side_points(box)
    theta = math.atan2(y1-y2, x2-x1)
    # 将弧度制的角度换算到一、四象限
    if theta > math.pi / 2.0:
        theta -= math.pi
    elif theta < -math.pi / 2.0:
        theta += math.pi
    angle = math.degrees(theta)
    return angle


def get_point_project_to_line(point, k, b):
    """
    计算点投影到直线的点的坐标
    :param point: 点的 x, y 坐标，point为元组类型
    :param k: 直线方程参数（斜率），直线方程为 y = k*x + b 形式
    :param b: 直线方程参数（截距）
    :return: 投影点浮点型坐标
    """
    x, y = point[0], point[1]
    x0 = (k*y + x - k*b) / (k*k + 1)
    y0 = k*x0 + b
    return x0, y0


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
    :return: 以左上角坐标点的x、y坐标拼接成的字符串
    """
    return "%d_%d" % (box[0][0], box[0][1])


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
    :param default_k: 当拟合直线的点数少于2个时，直线斜率使用默认斜率
    :return: 直线方程参数，k-斜率 b-截距
    """
    x_cors, y_cors = [], []
    draw_x1, draw_x2 = float("inf"), 0
    for box in boxes:
        ratio = calc_aspect_ratio(box)
        [x1, y1], [x2, y2] = get_side_points(box)
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
    if len(x_cors) >= 2:
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
            boxes = sorted(boxes, key=lambda t: t["box"][0][0])
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
    avg_line_gap, line_gap_sum = 0.0, 0.0  # 行与行之间距离的平均值、合计值
    line_split_fin = False  # 化验指标的行切割是否完毕
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
            if scale > threshold or get_top_horizon_overlap(box, line_boxes, threshold) is not None:
                # 对判定为同行的文本框从左到右排序
                line_boxes = sorted(line_boxes, key=lambda t: t["box"][0][0])
                line_gap = 0.0
                for t in line_boxes:
                    line_gap += t["dis"]
                line_gap /= len(line_boxes)
                if not line_split_fin:
                    # TODO: 化验单非化验结果行剔除阈值
                    if avg_line_gap != 0.0 and line_gap >= 2 * avg_line_gap:
                        line_split_fin = True
                    else:
                        key_word_cnt, tmp_key_words = 0, []
                        for t in line_boxes:
                            for w in t["seg_words"]:
                                if is_cfg_key_word(w):
                                    key_word_cnt += 1
                                    tmp_key_words.append(w)
                        if key_word_cnt > 0:
                            print("key_words in current line boxes, key_words=%s" % tmp_key_words)
                        else:
                            lines.append(line_boxes)
                            line_gap_sum += line_gap
                            avg_line_gap = line_gap_sum / len(lines)
                # 打印输出
                # print("[LINE %d]............................" % line_idx)
                pure_boxes = []
                for tmp_box in line_boxes:
                    _box = tmp_box["box"]
                    pure_boxes.append(_box)
                    # print("text=%s, dis=%.2f" % (tmp_box["text"], tmp_box["dis"]))
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
    :param head_line: 一维列表，[meta1, meta2, ...]
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
    boxes, seg_ratio_sum = [], 0.0
    total_word_cnt, word_set = 0.0, set()
    for meta in head_line:
        box = meta["box"]
        box_key = get_box_key(box)
        boxes.append(box)
        total_word_cnt += len(head_box_dict[box_key]["words"])
        char_count = 0
        for word in head_box_dict[box_key]["words"]:
            char_count += len(word)
            if word not in word_set:
                word_set.add(word)
        seg_ratio_sum += char_count * 1.0 / calc_valid_char(meta["text"])
        print("head_text=%s, head_words=%s, seg_words=%s" % (meta["text"], head_box_dict[box_key]["words"], meta["seg_words"]))
    seg_ratio = seg_ratio_sum / len(boxes)
    sub_table_ratio = total_word_cnt / len(word_set)
    print("total_word_cnt=%d, word_set=%s, sub_table_ratio=%.2f" % (total_word_cnt, word_set, sub_table_ratio))
    # TODO: 化验单栏数判断阈值设定
    if sub_table_ratio - math.floor(sub_table_ratio) >= 0.5:
        sub_table_cnt = int(math.ceil(sub_table_ratio))
    else:
        sub_table_cnt = int(math.floor(sub_table_ratio))
    # TODO: 化验单表头提取阈值设定
    if seg_ratio >= 0.75:
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
    line_box_dire_dict = {}  # {line_key: {box_key: dire, ...}, ...}
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


def box_horizon_overlap(box1, box2, mode="MAX"):
    """
    计算两个文本框在水平方向投影的交集，与两个文本框投影长度比例的较大值
    :param box1: 文本框1
    :param box2: 文本框2
    :param mode: 返回值取值模式，默认 "MAX"（两个比例中较大值），支持 "AVG"（两个比例的平均值）
    :return: 比例
    """
    lp_1, rp_1 = get_side_points(box1)
    lp_2, rp_2 = get_side_points(box2)
    x_cors = [(lp_1[0]+lp_2[0])/2, (rp_1[0]+rp_2[0])/2]
    y_cors = [(lp_1[1]+lp_2[1])/2, (rp_1[1]+rp_2[1])/2]
    k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    min_p, max_p = [None, None], [None, None]
    for i, box in enumerate([box1, box2]):
        for point in box:
            p = get_point_project_to_line(point, k, b)
            if min_p[i] is None or p[0] < min_p[i][0]:
                min_p[i] = p
            if max_p[i] is None or p[0] > max_p[i][0]:
                max_p[i] = p
    # min_p 中较大值
    lp = min_p[0] if min_p[0][0] >= min_p[1][0] else min_p[1]
    # max_p 中较小值
    rp = max_p[0] if max_p[0][0] <= max_p[1][0] else max_p[1]
    if lp[0] >= rp[0]:
        return 0
    # 两个文本框在直线上投影后，x方向交集的长度
    common_x_len = rp[0] - lp[0]
    # 文本框1、2在直线上投影后，x方向的长度
    x_len1 = max_p[0][0] - min_p[0][0]
    x_len2 = max_p[1][0] - min_p[1][0]
    ratio1 = common_x_len / x_len1
    ratio2 = common_x_len / x_len2
    if mode == "AVG":
        return (ratio1 + ratio2) / 2.0
    return ratio1 if ratio1 >= ratio2 else ratio2


def get_top_horizon_overlap(meta, meta_list, threshold=0.6, mode="MAX"):
    """
    获取一个文本框与文本框数组中有最大水平交集的文本框
    :param meta: 文本框
    :param meta_list: 文本框列表
    :param threshold: 阈值
    :param mode: 取投影交集的模式，支持 "MAX" 和 "AVG" 两种模式
    :return: 文本框
    """
    box = meta["box"]
    ans_meta, max_ratio = None, 0
    for tmp_meta in meta_list:
        tmp_box = tmp_meta["box"]
        ratio = box_horizon_overlap(box, tmp_box, mode=mode)
        if ratio > threshold and ratio > max_ratio:
            ans_meta = tmp_meta
            max_ratio = ratio
    return ans_meta


def points_to_line(points):
    """
    根据坐标点列表拟合直线方程
    :param points: 坐标点列表，[p1, p2, p3, ...]
    :return: 直线的斜率和截距
    """
    x_cors, y_cors = [], []
    for p in points:
        x_cors.append(p[0])
        y_cors.append(p[1])
    k, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    return k, b


def split_meta(box, seg_words):
    """
    分割表头文本框，例如"单位提示"分割为["单位", "提示"]，并以 word 长度占比分割 box
    :param box: 文本框坐标信息
    :param seg_words: 对文本框识别结果字符串，使用分词算法分割后的字符串列表
    :return: [meta1, meta2, ...]，meta 结构为 {"text": "单位", "box": [[(), (), (), ()], ...]}
    """
    if len(seg_words) == 1:
        return [{"text": seg_words[0], "box": box}]
    ratios = []
    total_sum, single_sum = 0.0, 0.0
    for w in seg_words:
        total_sum += str_len(w)
    for w in seg_words:
        single_sum += str_len(w)
        ratios.append(single_sum/total_sum)
    up_width = box[1][0] - box[0][0]
    down_width = box[2][0] - box[3][0]
    up_k, up_b = points_to_line([box[0], box[1]])
    down_k, down_b = points_to_line([box[2], box[3]])
    points = [(box[0], box[3])]
    for r in ratios:
        x = box[0][0] + up_width * r
        y = up_k * x + up_b
        p1 = (x, y)
        x = box[3][0] + down_width * r
        y = down_k * x + down_b
        p2 = (x, y)
        points.append((p1, p2))
    metas = []
    for i in range(1, len(points)):
        box = [points[i-1][0], points[i][0], points[i][1], points[i-1][1]]
        metas.append({"text": seg_words[i-1], "box": box})
    return metas


def merge_boxes(boxes):
    """
    合并文本框成一条线段，并取线段中间的一段
    :param boxes: 文本框列表
    :return: 线段的两个端点坐标
    """
    merged_box = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    for box in boxes:
        for i, p in enumerate(box):
            merged_box[i][0] += p[0]
            merged_box[i][1] += p[1]
    for p in merged_box:
        p[0] /= len(boxes)
        p[1] /= len(boxes)
    return merged_box


def split_vertical_lines(lines, head_line, head_box_dict):
    """
    在水平切分化验栏的基础上，再垂直切分化验栏。会修改入参lines中的meta信息，标记上所属表头类别，如meta["attrs"] = ["NO", "项目"]
    :param lines: 化验表中处于同一水平直线上的文本框，[meta1, meta2, ...]
    :param head_line: 表头文本框，[meta1, meta2, ...]
    :param head_box_dict: 表头文本框dict，box_key → {"seg_num": 3, "words": ["NO", "结果", "提示"]}
    :return: CSV表头列表，例如 ["NO", "项目", "结果", "单位", "参考范围"]
    """
    split_head_metas = []
    csv_head_words = []  # CSV表头
    for head_meta in head_line:
        box = head_meta["box"]
        box_key = get_box_key(box)
        seg_words = head_box_dict[box_key]["words"]
        csv_head_words.extend(seg_words)
        split_head_metas.extend(split_meta(box, seg_words))
    for head_meta in split_head_metas:
        candi_boxes = []
        for line in lines:
            # TODO: 候选文本框筛选阈值
            top_meta = get_top_horizon_overlap(head_meta, line, threshold=0.3, mode="AVG")
            if top_meta is not None:
                candi_boxes.append(top_meta["box"])
        if len(candi_boxes) == 0:
            continue
        # TODO: 剔除候选文本框异常值
        merged_box = merge_boxes(candi_boxes)
        merged_meta = {"box": merged_box}
        for line in lines:
            # TODO: 化验栏文本框分类的重叠阈值
            top_meta = get_top_horizon_overlap(merged_meta, line, threshold=0.5, mode="AVG")
            if top_meta is None:
                continue
            if "attrs" in top_meta:
                top_meta["attrs"].append(head_meta["text"])
            else:
                top_meta["attrs"] = [head_meta["text"]]
    split_and_correct(lines)
    return csv_head_words


def split_and_correct(lines):
    for line in lines:
        for meta in line:
            if "attrs" not in meta:
                continue
            source = meta["text"]
            attrs = meta["attrs"]
            categories = [get_category(attr) for attr in attrs]
            if len(categories) == 1:
                # correct single field
                res, _, _ = single_correct(source, category=categories[0])
                meta["corrected"] = [res]
            else:
                # correct multi field
                res = multi_correct(source, categories)
                meta["corrected"] = res
    return
