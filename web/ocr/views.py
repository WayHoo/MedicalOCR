import os
import time
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from .resp import get_suc_http_resp, get_fail_http_resp

from . import args, text_sys
import kie.structure.table_rec as table_rec
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from utils.img_process import imread_compress


@csrf_exempt
def user_login(request):
    if request.method == "GET":
        return get_fail_http_resp(info='request method not supported, use POST instead.')
    if request.user.is_authenticated:
        return get_suc_http_resp(info='already logged in, no need to log in again.')
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = authenticate(username=username, password=password)
    if user is not None:
        login(request, user)
        return get_suc_http_resp(info='login successfully.')
    else:
        return get_fail_http_resp(info='invalid login.')


@csrf_exempt
def user_logout(request):
    if request.method == "GET":
        return get_fail_http_resp(info='request method not supported, use POST instead.')
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = authenticate(username=username, password=password)
    if user is not None:
        logout(request)
        return get_suc_http_resp(info='logged out successfully.')
    else:
        return get_fail_http_resp(info='user is not logged in, no need to log out.')


@csrf_exempt
def upload_img(request):
    begin = time.time()
    if request.method != 'POST':
        return get_fail_http_resp(info='request method not supported, use POST instead.')
    if not request.user.is_authenticated:
        return get_fail_http_resp(info='log in first please.')
    # 获取上传的图片信息
    img = request.FILES.get('img', None)
    if img is None:
        return get_fail_http_resp(info='no image selected.')
    # 获取用户名
    username = request.POST.get('username', '')
    if username == '':
        return get_fail_http_resp(info='user name is not specified.')
    # 获取上传图片的名称
    img_name = img.name
    # 文件夹路径
    dir_path = os.path.join(settings.IMG_UPLOAD, username)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 图片保存路径
    img_path = os.path.join(dir_path, img_name)
    # 写入上传图片的内容
    with open(img_path, 'ab') as fp:
        # 如果上传的图片非常大，那么通过 img 对象的 chunks() 方法分割成多个片段来上传
        for chunk in img.chunks():
            fp.write(chunk)
    elapse = time.time() - begin
    print("process image cost %.3f" % elapse)
    # 化验单识别
    sheets, suc = ocr(img_path)
    if not suc:
        return get_fail_http_resp(info='ocr failed.')

    return get_suc_http_resp(data=sheets[0], info='upload successfully.')


def ocr(img_path):
    image_file = get_image_file_list(img_path)[0]
    try:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = imread_compress(image_file, compress=False)
        if img is None:
            print("error in loading image:{}".format(image_file))
            return None, False
        print("processing image:{}".format(image_file))
        start_time = time.time()
        dt_boxes, rec_res, img = text_sys(img)
        img_name = os.path.basename(image_file).split(".")[0]
        table_recog = table_rec.TableRecognizer(args, img, img_name, dt_boxes, rec_res)
        sheets = table_recog(save=False)
        elapse = time.time() - start_time
        print("process time of %s: %.3fs" % (image_file, elapse))
    except BaseException as e:
        print("Exception occurred: {}".format(e))
        return None, False
    return sheets, True
