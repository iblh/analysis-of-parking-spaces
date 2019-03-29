from xml.dom import minidom
from imutils import paths
from tqdm import tqdm
import numpy as np
import random
import json
import cv2
import os

# 初始化
target = 100
hst_len = 0
counter = 0
img_list = []
rd_seed = 628
pl_id = 'pucpr'
# pl_id = 'ufpr04'
# pl_id = 'ufpr05'
pbar = tqdm(total=target)
rootdir = './src_img/' + pl_id.upper()
hstdir = './train_data/train/' + pl_id + '/history.json'
# files_count = sum([len(files) for r, d, files in os.walk(rootdir)])

# 初始 history.json
hst_exists = os.path.isfile(hstdir)
if hst_exists:
    with open(hstdir) as json_file:
        hst_data = json.load(json_file)
        hst_len = len(hst_data['files'])
else:
    hst_data = {}
    hst_data['files'] = []


# 加载图片，提取图片路径
img_paths = sorted(list(paths.list_images(rootdir)))

# 伪随机图片路径 list
random.seed(rd_seed)
random.shuffle(img_paths)

# 循环图片路径
for img_path in img_paths:
    if counter == target:
        break

    # 初始化文件相关变量
    file = img_path.split(os.path.sep)[-1]
    img_date = file.split('.')[0]
    dir_name = file.split('_')[0]
    full_path = img_path.replace('.jpg', '')
    xml_exists = os.path.isfile(full_path + '.xml')

    if img_date in str(hst_data['files']):
        counter += 1
        pbar.update(1)
        continue

    # 解析 XML
    if xml_exists:
        xmldoc = minidom.parse(full_path + '.xml')
        spacelist = xmldoc.getElementsByTagName('space')
    else:
        print('ERROR: No XML File')
        continue

    # 循环处理单个停车位
    for space in spacelist:
        # print(img_date + ' ' + space.attributes['id'].value)
        if space.hasAttribute('occupied'):
            img = cv2.imread(full_path + '.jpg')
            status = int(space.attributes['occupied'].value)
            coordinate = []
            if len(space.getElementsByTagName('point')):
                points = space.getElementsByTagName('point')
            else:
                points = space.getElementsByTagName('Point')
        else:
            continue

        # 获取停车位 x&y 坐标
        for point in points:
            x = point.attributes['x'].value
            y = point.attributes['y'].value
            coordinate.append([int(x), int(y)])

        poly = np.array(coordinate)

        # 裁剪边界矩形 roi
        rect = cv2.boundingRect(poly)
        x, y, w, h = rect
        roi = img[y:y+h, x:x+w].copy()

        # 生成遮罩图层
        poly = poly - poly.min(axis=0)

        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.drawContours(mask, [poly], -1,
                         (255, 255, 255), -1, cv2.LINE_AA)

        # roi&mask 按位与运算
        dst = cv2.bitwise_and(roi, roi, mask=mask)

        # 添加白色背景
        bg = np.ones_like(roi, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst = bg + dst

        # 保存图片
        if status:
            cv2.imwrite('./train_data/train/' + pl_id + '/occupied/' +
                        space.attributes['id'].value + '-' + img_date + '.png', dst)
        else:
            cv2.imwrite('./train_data/train/' + pl_id + '/empty/' +
                        space.attributes['id'].value + '-' + img_date + '.png', dst)

    # 当前文件处理完成
    hst_data['files'].append({
        img_date: 1
    })
    counter += 1
    pbar.update(1)

pbar.close()

if target > hst_len:
    with open(hstdir, 'w') as outfile:
        json.dump(hst_data, outfile)
