import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from xml.dom import minidom

# 初始化
counter = 0
target = 2
pbar = tqdm(total=100)
rootdir = './src_img/PUCPR'
hstdir = './train_data/train/history.json'
# files_count = sum([len(files) for r, d, files in os.walk(rootdir)])

# 初始 history.json
hst_exists = os.path.isfile(hstdir)
if hst_exists:
    with open(hstdir) as json_file:
        hst_data = json.load(json_file)
else:
    hst_data = {}
    hst_data['files'] = []

for dirpath, dirnames, files in os.walk(rootdir):
    # 保留 .jpg 文件
    files = [file for file in files if file.endswith(('.jpg'))]
    for file in files:
        # print(img_date)
        if counter == target:
            break

        # 初始化文件相关变量
        img_date = file.split('.')[0]
        full_path = os.path.join(dirpath, file).replace('.jpg', '')
        xml_exists = os.path.isfile(full_path + '.xml')

        if img_date in str(hst_data['files']):
            continue

        if xml_exists:
            # 解析 XML
            xmldoc = minidom.parse(full_path + '.xml')
            spacelist = xmldoc.getElementsByTagName('space')
        else:
            pbar.update(100/target)
            continue

        for space in spacelist:
            if space.hasAttribute('occupied'):
                # print(space.attributes['id'].value)
                img = cv2.imread(full_path + '.jpg')
                status = int(space.attributes['occupied'].value)
                points = space.getElementsByTagName('point')
                coordinate = []
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
            output = bg + dst

            # 保存图片
            if status:
                cv2.imwrite('./train_data/train/occupied/pupcr-' +
                            space.attributes['id'].value + '-' + img_date + '.png', output)
            else:
                cv2.imwrite('./train_data/train/empty/pupcr-' +
                            space.attributes['id'].value + '-' + img_date + '.png', output)

        # 当前文件处理完成
        hst_data['files'].append({
            img_date: 1
        })
        pbar.update(100/target)
        counter += 1

pbar.close()
with open(hstdir, 'w') as outfile:
    json.dump(hst_data, outfile)
