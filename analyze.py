from keras.preprocessing.image import img_to_array
from keras.models import load_model
from xml.dom import minidom
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_date = '2012-09-11_15_16_58'
model_path = './train_data/v1.model'

img = np.array(Image.open(
    './test_images/' + img_date + '.jpg'), dtype=np.uint8)
fig, ax = plt.subplots(1, figsize=(12, 7.2))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                    wspace=0, hspace=0)

plt.axis('off')
plt.imshow(img)

# 解析 XML
xmldoc = minidom.parse('./test_images/' + img_date + '.xml')
spacelist = xmldoc.getElementsByTagName('space')


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(model_path)

for space in spacelist:
    # print(space.attributes['id'].value)
    points = space.getElementsByTagName('point')
    coordinate = []
    x = y = 0

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
    roi = img[y:y+h, x:x+w].copy()

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
    image = bg + dst

    # 将图片进行预处理用以下一步识别
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # 使用 model 识别停车位状态
    (empty, occupied) = model.predict(image)[0]

    # 创建状态和概率 name
    status = 1 if occupied > empty else 0
    proba = occupied if occupied > empty else empty
    proba = '{:.2f}'.format(proba)

    plt.text(x + 10, y + 10, proba, fontsize=8,
             bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 2})

    # 设置停车位边缘，添加 patch 到 axes
    if status:
        patches_poly = patches.Polygon(
            np.array(coordinate), fill=False, color='#c40b13', linewidth=1.5)
    else:
        patches_poly = patches.Polygon(
            np.array(coordinate), fill=False, color='#18A309', linewidth=1.5)

    ax.add_patch(patches_poly)


plt.show()
