from keras.preprocessing.image import img_to_array
from keras.models import load_model
from xml.dom import minidom
from imutils import paths
from tqdm import tqdm
import numpy as np
import argparse
import random
import cv2
import os

# 构造参数处理
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#                 help="path to model")
# ap.add_argument("-p", "--plid", required=True,
#                 help="parking lot id")
# args = vars(ap.parse_args())


def main(pl_id, target, model, pbar):
    global proba_list
    counter = 0
    hit_num = 0
    rd_seed = 628
    dims = (40, 40, 3)
    rootdir = './src_img/' + pl_id.upper()

    # 加载图片，提取图片路径
    img_paths = sorted(list(paths.list_images(rootdir)))

    # 伪随机图片路径 list，并逆转以区分训练数据
    random.seed(rd_seed)
    random.shuffle(img_paths)
    img_paths.reverse()

    # 循环图片路径
    for img_path in img_paths:
        # 初始化文件相关变量
        # breaking = False
        full_path = img_path.replace('.jpg', '')
        xml_exists = os.path.isfile(full_path + '.xml')

        # 解析 XML
        if xml_exists:
            bgr_img = cv2.imread(full_path + '.jpg')
            xmldoc = minidom.parse(full_path + '.xml')
            spacelist = xmldoc.getElementsByTagName('space')
        else:
            print('ERROR: No XML File')
            continue

        for space in spacelist:
            if counter == target:
                proba_list.append(hit_num / counter * 100)
                return

            # print(space.attributes['id'].value)
            if space.hasAttribute('occupied'):
                x = y = 0
                status = int(space.attributes['occupied'].value)
                coordinate = []

                if len(space.getElementsByTagName('point')):
                    points = space.getElementsByTagName('point')
                else:
                    points = space.getElementsByTagName('Point')
            else:
                continue

            for point in points:
                # 停车位 x&y 坐标
                x = int(point.attributes['x'].value)
                y = int(point.attributes['y'].value)
                coordinate.append([x, y])

            array_poly = np.array(coordinate)

            # 提取识别 roi
            # 裁剪边界矩形 roi
            rect = cv2.boundingRect(array_poly)
            x, y, w, h = rect
            roi = bgr_img[y:y+h, x:x+w].copy()

            # 生成遮罩图层
            array_poly = array_poly - array_poly.min(axis=0)

            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(mask, [array_poly], -1,
                             (255, 255, 255), -1, cv2.LINE_AA)

            # roi&mask 按位与运算
            dst = cv2.bitwise_and(roi, roi, mask=mask)

            # 添加白色背景
            bg = np.ones_like(roi, np.uint8)*255
            cv2.bitwise_not(bg, bg, mask=mask)
            dst = bg + dst

            # 将图片进行预处理用以下一步识别
            image = cv2.resize(dst, (dims[1], dims[0]))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # 使用 model 识别停车位状态
            (empty, occupied) = model.predict(image)[0]

            # 创建状态和概率 name
            pd_status = 1 if occupied > empty else 0
            if status == pd_status:
                hit_num += 1

            counter += 1
            pbar.update(1)


# 初始化
target = 20000
pl_list = ['pucpr', 'ufpr04', 'ufpr05']
proba_list = []
model_path = './train_data/models/tinyvgg-200.model'

# 加载训练完成的 CNN 网络模型
print("[INFO] loading network...")
model = load_model(model_path)

if os.name == 'nt':
    pbar = tqdm(total=target*len(pl_list), ascii=True)
else:
    pbar = tqdm(total=target*len(pl_list))

for pl_id in pl_list:
    main(pl_id, target, model, pbar)

pbar.close()

for i, proba in enumerate(proba_list):
    print('{}: {:.2f}%'.format('Accuracy of ' + pl_list[i].upper(), proba))
