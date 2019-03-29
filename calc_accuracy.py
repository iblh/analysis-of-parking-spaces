from keras.preprocessing.image import img_to_array
from keras.models import load_model
from xml.dom import minidom
from imutils import paths
from tqdm import tqdm
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# 初始化
hits = 0
target = 100
ps_conter = 0
img_counter = 0
rd_seed = 628
pl_id = 'pucpr'
dims = (40, 40, 3)
pbar = tqdm(total=target)
rootdir = './src_img/' + pl_id.upper()
model_path = './train_data/tinyvgg-pucpr-100.model'

# 加载图片，提取图片路径
img_paths = sorted(list(paths.list_images(rootdir)))
# img_paths = ['./src_img/PUCPR/2012-09-11/2012-09-11_15_16_58.jpg']

# 加载训练完成的 CNN 网络模型
print("[INFO] loading network...")
model = load_model(model_path)

# 伪随机图片路径 list，并逆转
random.seed(rd_seed)
random.shuffle(img_paths)
img_paths.reverse()

# 循环图片路径
for img_path in img_paths:
    if img_counter == target:
        break

    # 初始化文件相关变量
    file = img_path.split(os.path.sep)[-1]
    img_date = file.split('.')[0]
    dir_name = file.split('_')[0]
    full_path = img_path.replace('.jpg', '')
    xml_exists = os.path.isfile(full_path + '.xml')

    # 解析 XML
    if xml_exists:
        bgr_img = cv2.imread(full_path + '.jpg')
        xmldoc = minidom.parse(full_path + '.xml')
        spacelist = xmldoc.getElementsByTagName('space')

        # 初始化 plt 图像
        # rgb_img = np.array(Image.open(full_path + '.jpg'))
        # fig, ax = plt.subplots(1, figsize=(15, 9))
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
        #                     wspace=0, hspace=0)
        # plt.imshow(rgb_img)
        # plt.axis('off')
    else:
        print('ERROR: No XML File')
        continue

    for space in spacelist:
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
            hits += 1

        ps_conter += 1

        # 设置停车位边缘，添加 patch 到 axes
        # if pd_status:
        #     patches_poly = patches.Polygon(
        #         np.array(coordinate), fill=False, color='#c40b13', linewidth=1.5)
        # else:
        #     patches_poly = patches.Polygon(
        #         np.array(coordinate), fill=False, color='#18A309', linewidth=1.5)

        # ax.add_patch(patches_poly)

    img_counter += 1
    pbar.update(1)
    # plt.show()


pbar.close()
print('{}: {:.2f}%'.format('Accuracy', hits / ps_conter * 100))
