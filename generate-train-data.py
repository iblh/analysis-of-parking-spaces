import os
import cv2
import numpy as np
from xml.dom import minidom
from tqdm import tqdm

rootdir = './src_img/PUCPR'

pbar = tqdm(total=100)
files_count = sum([len(files) for r, d, files in os.walk(rootdir)])

for dirpath, dirnames, files in os.walk(rootdir):
    files = [file for file in files if file.endswith(('.jpg'))]
    # print('111')
    for file in files:
        # print(img_date)
        img_date = file.split('.')[0]
        full_path = os.path.join(dirpath, file).replace('.jpg', '')
        exists = os.path.isfile(full_path + '.xml')

        if exists:
            # 解析 XML
            xmldoc = minidom.parse(full_path + '.xml')
            spacelist = xmldoc.getElementsByTagName('space')
        else:
            pbar.update(100/files_count*2)
            continue

        for space in spacelist:
            if space.hasAttribute('occupied'):
                # print(space.attributes['id'].value)
                img = cv2.imread(full_path + '.jpg')
                status = int(space.attributes['occupied'].value)
                points = space.getElementsByTagName('point')
                coordinate = []

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

        pbar.update(100/files_count*2)

pbar.close()
