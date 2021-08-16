import cv2
import time
import numpy as np
from ppocr.utils.utility import get_image_file_list

# reference: https://www.coder.work/article/2087730


def detect_angle(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if 45000 > area > 20:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = mask.shape

    # Horizontal
    if w > h:
        left = mask[0:h, 0:0 + w // 2]
        right = mask[0:h, w // 2:]
        left_pixels = cv2.countNonZero(left)
        right_pixels = cv2.countNonZero(right)
        return 0 if left_pixels >= right_pixels else 180
    # Vertical
    else:
        top = mask[0:h // 2, 0:w]
        bottom = mask[h // 2:, 0:w]
        top_pixels = cv2.countNonZero(top)
        bottom_pixels = cv2.countNonZero(bottom)
        return 90 if bottom_pixels >= top_pixels else 270


if __name__ == '__main__':
    image_dir = "./doc/imgs/test_sheets/batch_001/"
    image_file_list = get_image_file_list(image_dir)
    total = 0
    err_num = 0
    begin = time.time()
    for image_file in image_file_list:
        image = cv2.imread(image_file)
        if image is None:
            continue
        total += 1
        angle = detect_angle(image)
        if angle != 0:
            err_num += 1
        print("%s: %d" % (image_file, angle))
    end = time.time()
    print("total=%d, err_num=%d" % (total, err_num))
    print("accuracy=%.2f%s" % ((total-err_num)*100/total, "%"))
    print("total time cost=%.2fs" % (end - begin))
    print("time cost per image=%.2fs" % ((end - begin) / total))
