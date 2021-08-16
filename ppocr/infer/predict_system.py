import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import cv2
import copy
import numpy as np
import time
from PIL import Image
import ppocr.infer.utility as utility
import ppocr.infer.predict_rec as predict_rec
import ppocr.infer.predict_det as predict_det
import ppocr.infer.predict_cls as predict_cls
import ppocr.infer.predict_agl as predict_agl
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from ppocr.infer.utility import draw_ocr_box_txt
from ppocr.utils.block_process import calc_block_angle, block_seg, extract_test_sheet
from utils.img_process import imread_compress

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_det = args.use_angle_det
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)
        if self.use_angle_det:
            self.angle_detector = predict_agl.AngleDetector(args)

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        rotated = False  # whether has it been rotated
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)  # rotate 90° counterclockwise
            rotated = True
        return dst_img, rotated

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        save_path = "./output/inference_results/test_sheets/angle/"
        if self.use_angle_det:
            angle, img = self.angle_detector(img)
            # img_path = os.path.join(save_path, "agl_det_res.jpg")
            # cv2.imwrite(img_path, img)

        round_idx, max_round = 0, 2
        while round_idx < max_round:
            round_idx += 1
            ori_im = img.copy()
            dt_boxes, elapse = self.text_detector(img)
            logger.info("dt_boxes num: {}, elapse: {}".format(len(dt_boxes), elapse))
            if dt_boxes is None:
                return None, None
            img_crop_list = []
            det_rotate_list = []
            dt_boxes = sorted_boxes(dt_boxes)

            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop, rotated = self.get_rotate_crop_image(ori_im, tmp_box)
                img_crop_list.append(img_crop)
                det_rotate_list.append(rotated)
            if self.use_angle_cls:
                img_crop_list, angle_list, cls_rotate, elapse = self.text_classifier(
                    img_crop_list, det_rotate_list)
                logger.info("cls num: {}, cls_rotate: {}°, elapse: {}".
                            format(len(img_crop_list), cls_rotate, elapse))
                if cls_rotate != 0:
                    rotate_param = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}
                    img = cv2.rotate(img, rotate_param[cls_rotate])
                    # img_path = os.path.join(save_path, "agl_det_cls_res.jpg")
                    # cv2.imwrite(img_path, img)
                    continue

            rec_res, elapse = self.text_recognizer(img_crop_list)
            logger.info("rec_res num: {}, elapse: {}".format(len(rec_res), elapse))
            # self.print_draw_crop_rec_res(img_crop_list, rec_res)
            filter_boxes, filter_rec_res = [], []
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                text, score = rec_reuslt
                if score >= self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)
            return filter_boxes, filter_rec_res, img


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    # for i in range(num_boxes - 1):
    #     if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
    #             (_boxes[i + 1][0][0] < _boxes[i][0][0]):
    #         tmp = _boxes[i]
    #         _boxes[i] = _boxes[i + 1]
    #         _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    error_img_list = []
    for image_file in image_file_list:
        try:
            img, flag = check_and_read_gif(image_file)
            if not flag:
                # TODO: image compress
                img = imread_compress(image_file, compress=False)
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            logger.info("processing image:{}".format(image_file))
            start_time = time.time()
            dt_boxes, rec_res, img = text_sys(img)
            # 空跑 GPU
            while args.use_gpu and args.grab_gpu:
                time.sleep(1000)
            elapse = time.time() - start_time
            logger.info("Predict time of %s: %.3fs" % (image_file, elapse))
            # print('dt_boxes=%s' % dt_boxes)
            # calc_block_angle(dt_boxes, rec_res)
            # post_process_img = block_seg(img, dt_boxes)
            extract_test_sheet(img, args, os.path.basename(image_file).split(".")[0], dt_boxes, rec_res)

            # out = []
            # import json
            # for i in range(len(rec_res)):
            #     text, score = rec_res[i]
            #     points = []
            #     for box in dt_boxes[i]:
            #         m, n = box
            #         l = []
            #         l.append(float(m))
            #         l.append(float(n))
            #         points.append(l)
            #     tmp = {}
            #     tmp["text"] = text
            #     tmp["score"] = float(score)
            #     tmp["points"] = points
            #     out.append(tmp)
            # print(json.dumps(out))

                # logger.info("{}, {:.3f}, {}".format(text, score, dt_boxes[i]))

            # for text, score in rec_res:
            #     logger.info("{}, {:.3f}".format(text, score))

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path)
                # draw_img_save = "./output/inference_results/"
                draw_img_save = args.save_path
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename(image_file)),
                    draw_img[:, :, ::-1])
                # post_process_img_save = "./output/post_process_results/"
                # if not os.path.exists(post_process_img_save):
                #     os.makedirs(post_process_img_save)
                # cv2.imwrite(
                #     os.path.join(post_process_img_save, os.path.basename(image_file)),
                #     post_process_img[:, :, ::-1])
                logger.info("The visualized image saved in {}".format(
                    os.path.join(draw_img_save, os.path.basename(image_file))))
        except BaseException as e:
            error_img_list.append(image_file)
            logger.error("Exception occurred: {}".format(e))
            continue
    print('----------------image process statistic----------------')
    print('total image num:', len(image_file_list))
    print('error image num:', len(error_img_list))
    print('error image list:', error_img_list)


if __name__ == "__main__":
    begin = time.time()
    main(utility.parse_args())
    end = time.time()
    print("predict system execute time: %.2fs" % (end - begin))
