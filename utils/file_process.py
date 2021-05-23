import os
from ppocr.utils.utility import get_image_file_list


def batch_rename_img(dir_path, name_prefix):
    img_list = get_image_file_list(dir_path)
    size = len(str(len(img_list)))
    for i, img_file in enumerate(img_list):
        suffix = str(i+1).zfill(size)
        new_name = os.path.split(img_file)[0] + '/' + name_prefix + "_" + suffix + os.path.splitext(img_file)[1]
        print('------------FILE %d------------' % (i+1))
        print('pre_img_name:', img_file)
        print('new_img_name:', new_name)
        os.rename(img_file, new_name)


if __name__ == '__main__':
    batch_rename_img('../doc/imgs/other_test_sheet/', 'other_test_sheet')
