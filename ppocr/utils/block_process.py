# coding=utf-8
import math
import time
import cv2
import pandas as pd
import numpy as np
from scipy import optimize, stats
from utils.nlp import jieba_seg
from PIL import Image, ImageDraw, ImageFont


def calc_block_angle(dt_boxes, rec_res):
    '''
    根据文本检测框计算文本倾斜角度
    :param dt_boxes: 文本检测框列表
    [array([[262., 130.], [423., 130.], [423., 167.], [262., 167.]], dtype=float32), ...]
    :param rec_res: 文本块内容列表
    :return: 倾斜角度（弧度）
    '''

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


# 化验单内容提取
def test_sheet_extract(img, dt_boxes, rec_res):
    # 化验栏标题
    with open("./doc/dict/test_sheet_dict.txt", "r", encoding="utf-8") as f:
        head_words = f.read().splitlines()
    text_map = {}
    # 候选表头文本框，候选非表头文本框
    candi_head_boxes, candi_other_boxes = [], []
    head_box_dict = {}
    begin = time.time()
    print("jieba_seg begin...")
    for i in range(len(dt_boxes)):
        text, score = rec_res[i]
        box = dt_boxes[i]
        box_key = get_box_key(box)
        meta = {'text': text, 'score': score, 'box': box}
        # ---------------------jieba_seg-----------------------
        words = jieba_seg(text)
        for word in words:
            if word in head_words:
                if box_key not in head_box_dict.keys():
                    head_box_dict[box_key] = [word]
                    candi_head_boxes.append(meta)
                else:
                    head_box_dict[box_key].append(word)
                print("word=%s, text=%s, box=%s" % (word, text, box))
        # ---------------------jieba_seg-----------------------
        if text not in text_map:
            text_map[text] = [meta]
        else:
            text_map[text].append(meta)
    end = time.time()
    print("jieba_seg end, cost %.2fs" % (end - begin))

    # 提取化验栏标题并判断单栏或双栏
    head_boxes, other_boxes = [], []
    diff_word_cnt, total_word_cnt = 0.0, 0.0
    for key in text_map:
        word = str(key).upper().strip().strip('.')
        if word in head_words:
            diff_word_cnt += 1
            total_word_cnt += len(text_map[key])
            head_boxes.extend(text_map[key])
        else:
            other_boxes.extend(text_map[key])

    # TODO: 分词方式提取化验栏标题，可处理竖直方向多个化验表，水平方向不支持
    head_lines = extract_head_lines(candi_head_boxes, head_box_dict)


    # TODO: 一张图片可能包含多张化验单，需要对table_heads做聚类处理
    ratio = total_word_cnt / diff_word_cnt
    # TODO: ratio阈值用于判断双栏或单栏化验单，经验值设定
    if ratio > 1.2:
        print('双栏化验单')
    else:
        print('单栏化验单')
    print('head_word_ratio=%f' % ratio)

    x_cors, y_cors = [], []
    # 根据文本框左上角坐标点的横坐标从小到大排序（文本框从左到右排序）
    head_boxes = sorted(head_boxes, key=lambda head: head['box'][0][0])
    for item in head_boxes:
        box = item['box']
        x_cors.extend([(box[0][0] + box[3][0]) / 2.0, (box[1][0] + box[2][0]) / 2.0])
        y_cors.extend([(box[0][1] + box[3][1]) / 2.0, (box[1][1] + box[2][1]) / 2.0])
        print('text=%s, score=%s, box=%s' % (item['text'], item['score'], box))
    # 直线拟合
    # TODO: 剔除坐标点中的异常值
    a, b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
    # TODO: 处理表头直线穿过的所有文本框
    new_other_boxes = []
    for item in other_boxes:
        box = item['box']
        if is_box_on_line(box, a, b):
            pass
        else:
            new_other_boxes.append(item)
    other_boxes = new_other_boxes



    # TODO: 根据表头限定化验栏在x方向的范围，过于依赖表头处理的准确度
    x_min, x_max = x_cors[0], x_cors[-1]
    y_min, y_max = y_cors[0], y_cors[-1]
    # 直线绘制
    # img.shape (height, width, channel)
    x1, y1 = int(x_min), int(f_1(x_min, a, b))
    x2, y2 = int(x_max), int(f_1(x_max, a, b))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # 计算文本块到直线的距离
    up_boxes = []  # 化验单表头之上文本框
    down_boxes = []  # 化验单表头之下文本框
    vertical_boxes = []  # 竖直的文本框
    ratio_dict = {}
    for item in other_boxes:
        box = item["box"]
        # 文本块宽高比
        ratio = calc_width_height_ratio(box)
        ratio_dict[box[0][0]] = ratio
        # 文本倾斜角
        angle = calc_box_angle(box)
        if ratio < 0.5 or angle > 45 or angle < -45:
            # 剔除竖向文本框
            vertical_boxes.append(item)
            continue
        x, y = calc_center_point(item["box"])
        dis = calc_point_to_line(a, -1, b, x, y)
        item["dis"] = dis
        y0 = f_1(x, a, b)
        if y0 < y:
            down_boxes.append(item)
        else:
            up_boxes.append(item)
    # up_boxes = sorted(up_boxes, key=lambda t: t["dis"], reverse=True)
    down_boxes = sorted(down_boxes, key=lambda t: t["dis"])
    # print("-------------up_boxes--------------")
    # pre_val = 0
    # for box in up_boxes:
    #     delta = box["dis"] - pre_val
    #     pre_val = box["dis"]
    #     height = (box["box"][2][1] + box["box"][3][1] - box["box"][0][1] - box["box"][1][1]) / 2.0
    #     print("text=%s, score=%.1f, dis=%.1f, delta=%.1f, height=%.1f" % (box["text"], box["score"], box["dis"], delta, height))
    print("-------------down_boxes--------------")
    line_idx = 0
    pre_k = a  # 前一条直线斜率
    scale_threshold = 0.6
    while True:
        if len(down_boxes) == 0:
            break
        line_idx += 1  # 行号
        box_num = 0  # 同一行文本框中的编号
        # height_sum为同一行文本框的y方向高度累加值
        height_sum, delta_sum = 0.0, 0.0
        # avg_height为判定为同一行的文本框的平均高度；
        # avg_delta为判定为同一行的文本框中，相邻文本框到直线距离的差值的平均值
        avg_height, avg_delta = 0.0, 0.0
        pre_val = down_boxes[0]["dis"]
        line_boxes = []
        for idx, box in enumerate(down_boxes):
            delta = box["dis"] - pre_val
            scale = 0
            # 使用当前文本框与前一个文本框到直线距离的差值 占 前面同一行文本框高度平均值的比例 来判断当前文本框是否属于当前行
            if avg_height != 0:
                scale = delta / avg_height
            # 边界case，最后一个box
            if scale < scale_threshold and idx + 1 == len(down_boxes):
                line_boxes.append(box)
                scale = 1
            if scale >= scale_threshold:
                # 对判定为同行的文本框从左到右排序
                line_boxes = sorted(line_boxes, key=lambda t: t["box"][0][0])
                # 打印输出
                print("[LINE %d]............................" % line_idx)
                x_cors, y_cors = [], []
                for tmp_box in line_boxes:
                    _box = tmp_box["box"]
                    # 文本块宽高比
                    ratio = ratio_dict[_box[0][0]]
                    if ratio < 1:
                        x, y = calc_center_point(_box)
                        x_cors.append(x)
                        y_cors.append(y)
                    else:
                        x_cors.extend([(_box[0][0] + _box[3][0]) / 2.0, (_box[1][0] + _box[2][0]) / 2.0])
                        y_cors.extend([(_box[0][1] + _box[3][1]) / 2.0, (_box[1][1] + _box[2][1]) / 2.0])
                    print("text=%s, dis=%.2f, ratio=%.2f" % (tmp_box["text"], tmp_box["dis"], ratio_dict[_box[0][0]]))
                # 当一行的文本框不少于2个时，才重新计算该行直线方程；否则沿用上一行的直线斜率
                if len(x_cors) > 2:
                    # 确定该行的直线方程并绘制直线
                    _a, _b = optimize.curve_fit(f_1, x_cors, y_cors)[0]
                    pre_k = _a
                else:
                    # 文本框的坐标重心
                    tmp_x, tmp_y = np.mean(x_cors), np.mean(y_cors)
                    _a = pre_k
                    _b = tmp_y - _a * tmp_x
                # 绘制当前行有文本框的范围
                x1, x2 = int(x_cors[0]), int(x_cors[-1])
                y1, y2 = int(f_1(x_cors[0], _a, _b)), int(f_1(x_cors[-1], _a, _b))
                # 仅绘制表头宽度范围，严重依赖于表头处理，bad case较多
                # x1, y1 = calc_point_project_to_line((x_min, y_min), _a, _b)
                # x2, y2 = calc_point_project_to_line((x_max, y_max), _a, _b)
                # 绘制整个图片宽度范围
                # x1, x2 = 0, img.shape[1]
                # y1, y2 = int(f_1(x1, _a, _b)), int(f_1(x2, _a, _b))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
                # 计算剩余文本框到该直线的距离
                down_boxes = down_boxes[len(line_boxes):]
                if len(down_boxes) == 0:
                    break
                for _box in down_boxes:
                    x, y = calc_center_point(_box["box"])
                    dis = calc_point_to_line(_a, -1, _b, x, y)
                    _box["dis"] = dis
                down_boxes = sorted(down_boxes, key=lambda t: t["dis"])
                break
            else:
                # 没有换行，继续累加文本框
                box_num += 1
                line_boxes.append(box)
                delta_sum += delta
                height = (box["box"][2][1] + box["box"][3][1] - box["box"][0][1] - box["box"][1][1]) / 2.0
                height_sum += height
                avg_height = height_sum / box_num
                avg_delta = delta_sum / box_num
                pre_val = box["dis"]
                # print("text=%s, dis=%.1f, delta=%.2f, avg_delta=%.2f, avg_height=%.2f" % (box["text"], box["dis"], delta, avg_delta, avg_height))

    # up_downs = []
    # up_down_words = {'姓名', '年龄', '性别', '住院号', '病历号', '病人号', '病案号', '病人类型', '患者类别',
    #                  '科室', '科别', '申请科室', '临床诊断', '报告单号', '验单号', '检验单号', '床号', '病床号',
    #                  '病区', '标本种类', '标本类型', '标本', '样本类型', '样本种类', '检验标本', '检验项目',
    #                  '样本类型', '标本号', '样本号', '标本编号', '流水号', '标本说明', '备注', '声明', '诊断',
    #                  '样本状态', '费别', '申请医生', '送检医生', '送检者', '采样时间', '送检日期', '送检时间',
    #                  '接收时间', '报告时间', '报告日期', '打印时间', '检验者', '检验员', '操作员', '检验医师',
    #                  '审核', '复核员', '审核者', '审核医师', '核对者', '审核人', '签字', '接收者', '审核时间',
    #                  '检验日期', '检验时间', '报告人'}
    # for word in up_down_words:
    #     if word in text_map:
    #         up_downs.extend(text_map[word])

    # print('-------------up-down-box-------------')
    # up_downs = sorted(up_downs, key=lambda x: (x['box'][0][1], x['box'][0][0]))
    # for item in up_downs:
    #     print('text=%s, score=%s, box=%s' % (item['text'], item['score'], item['box']))
    return a, b


def get_middle_point(box):
    x, y = 0.0, 0.0
    for point in box:
        x += point[0]
        y += point[1]
    return x / 4.0, y / 4.0


# 计算点到直线的距离，直线方程 a*x + b*y + c = 0
def calc_point_to_line(a, b, c, x, y):
    return math.fabs(a * x + b * y + c) / math.sqrt(a * a + b * b)


# 计算矩形文本框的质心坐标
def calc_center_point(box):
    x = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
    y = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0
    return x, y


# 直线方程函数
def f_1(x, a, b):
    return a * x + b


# 计算文本框的高宽比
def calc_width_height_ratio(box):
    width = (box[1][0] + box[2][0] - box[0][0] - box[3][0]) / 2.0
    height = (box[2][1] + box[3][1] - box[0][1] - box[1][1]) / 2.0
    ratio = math.fabs(width / height)
    return ratio


# 计算文本框倾斜角
def calc_box_angle(box):
    x1, x2 = (box[0][0] + box[3][0]) / 2.0, (box[1][0] + box[2][0]) / 2.0
    y1, y2 = (box[0][1] + box[3][1]) / 2.0, (box[1][1] + box[2][1]) / 2.0
    theta = math.atan2(y1 - y2, x2 - x1)
    # 将弧度制的角度换算到一、四象限
    if theta > math.pi / 2.0:
        theta -= math.pi
    elif theta < -math.pi / 2.0:
        theta += math.pi
    angle = math.degrees(theta)
    return angle


def calc_point_project_to_line(point, a, b):
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


def is_box_on_line(box, a, b):
    """
    判断直线是否穿过文本框
    :param box: 文本框
    :param a: 直线方程参数（斜率），直线方程为 y = ax + b 形式
    :param b: 直线方程参数（截距）
    :return: True or False
    """
    flag_list = []
    for point in box:
        x0, y0 = point[0], point[1]
        y = a*x0 + b
        flag = 1 if y0 >= y else -1
        flag_list.append(flag)
    res = 1
    for flag in flag_list:
        res *= flag
    return True if res <= 0 else False


def get_box_key(box):
    """
    获取文本框的hash key，用于dict
    :param box: 文本框
    :return: 以1、4坐标点的x、y坐标拼接成的字符串
    """
    return "%d_%d_%d_%d" % (box[0][0], box[0][1], box[3][0], box[3][0])


def extract_head_lines(candi_head_boxes, head_box_dict):
    for k, v in candi_head_boxes.items():
        box = v["box"]
        # 确定文本框的直线方程
