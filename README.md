# MedicalOCR
## Step 1: Build appropriate environment
```commandline
# install appropriate paddlepaddle by yourself
# https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html
conda create -n medical_ocr python=3.7
pip install -r requirements.txt
```
Note: If you have something wrong with "from shapely.geometry import Polygon"，uninstall shapely and install appropriate shapely version using whl package. Reference https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely

## Step 2: Download pre-trained inference model
```commandline
# download detection pre-trained inference model
cd MedicalOCR/inference
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar && tar -xvf ch_ppocr_server_v2.0_det_infer.tar

# download direction classifier pre-trained inference model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

# download recognition pre-trained inference model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar && tar -xvf ch_ppocr_server_v2.0_rec_infer.tar
```

# Web
## Step 1: Django migrate
```commandline
cd MedicalOCR
export PYTHONPATH=$PYTHONPATH:$(pwd)
python web/manage.py makemigrations
python web/manage.py migrate
```

## Step 2: create super user
```commandline
python web/manage.py createsuperuser
# Then input user name, email and password
```

## Step 3: Run server
```commandline
python web/manage.py runserver
```