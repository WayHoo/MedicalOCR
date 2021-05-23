import cv2
import numpy as np
import os
import math
import time


# 直线方程函数
def f_1(x, a, b):
    return a * x + b


def auto_canny(image, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 计算单通道像素强度的中位数
    mid = np.median(gray)
    # 选择合适的lower和upper值，然后应用它们
    lower = int(max(0, (1.0 - sigma) * mid))
    upper = int(min(255, (1.0 + sigma) * mid))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3)  # apertureSize参数默认其实就是3
    cv2.imwrite(os.path.join("./output/post_process_results/", "edges.jpg"), edges)
    return edges


def line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# 统计概率霍夫线变换(效果不佳)
def hough_p_line_detect(image):
    min_w_h = min(image.shape[0], image.shape[1])
    begin = time.process_time()
    edges = auto_canny(image)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    end = time.process_time()
    print("hough_p_line_detect execute time: %.2fs" % (end - begin))
    for line in lines:
        length = line_length(line[0])
        if length >= 0.1 * min_w_h:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join("./output/post_process_results/", "hough_p_line.jpg"), image)


def fld_line_detect(image):
    min_w_h = min(image.shape[0], image.shape[1])
    begin = time.process_time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 创建一个FLD对象
    fld = cv2.ximgproc.createFastLineDetector()
    # 执行检测结果
    lines = fld.detect(gray)
    # 对lines从上到下、从左到右排序
    tmp_lines = []
    for line in lines:
        line = line[0]
        length = line_length(line)
        if length < 0.1 * min_w_h:
            continue
        if line[0] > line[2]:
            print("swap line point...")
            line = [line[2], line[3], line[0], line[1]]
        tmp_lines.append(line)
    lines = sorted(tmp_lines, key=lambda l: (l[1], l[0]))
    end = time.process_time()
    print("fld_line_detect execute time: %.2fs" % (end - begin))
    rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    i = 0
    for line in lines:
        length = line_length(line)
        x1, y1, x2, y2 = line
        slope = math.degrees(math.atan2(y2 - y1, x2 - x1))
        print("line %d: %s, len=%.2f, slope=%.2f°" % (i + 1, line, length, slope))
        # print("[%f, %f, %f, %f]" % (line[0], line[1], line[2], line[3]))
        color = rgb[i % len(rgb)]
        cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        i += 1
    # if i == 9:
    # 	break

    # 绘制检测结果
    cv2.imwrite(os.path.join("./output/post_process_results/", "fld_line.jpg"), image)


# 压缩图片文件
def imread_compress(img_path, compress=True):
    # img_res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # cv2.imwrite(os.path.join("./output/post_process_results/", "img_resize.jpg"), img_res)
    # return img_res
    begin = time.time()
    o_size = os.path.getsize(img_path)
    print("[Origin] image memory size = %.2fKB" % (o_size / 1024.0))
    img = cv2.imread(img_path)
    scale = int(51200 * 1024.0 / o_size)
    if compress and scale < 100:
        save_path = "./output/post_process_results/img_resize.jpg"
        cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), scale])
        img = cv2.imread(save_path)
        print("[Compressed] image memory size = %.2fKB" % (os.path.getsize(save_path) / 1024.0))
    end = time.time()
    print("image read and compress execute time: %.2fs" % (end - begin))
    return img


if __name__ == "__main__":
    img_path = r"./doc/imgs/check_report_07.jpg"
    src = cv2.imread(img_path)
    print("image shape =", src.shape)
    # hough_p_line_detect(src.copy())
    # auto_canny(src)
    # fld_line_detect(src.copy())
    img_res = imread_compress(img_path, compress=True)
    print("image shape after resize =", img_res.shape)
