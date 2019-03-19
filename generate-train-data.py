import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
from xml.dom import minidom

img_date = '2012-10-12_06_27_31'

# 解析 XML
xmldoc = minidom.parse('./test_images/' + img_date + '.xml')
spacelist = xmldoc.getElementsByTagName('space')

print(len(spacelist))

for space in spacelist:
    # read image as RGB and add alpha (transparency)
    img = cv2.imread('./test_images/' + img_date + '.jpg')

    points = space.getElementsByTagName('point')

    coordinate = []

    for point in points:

        # 停车位 x&y 坐标
        x = point.attributes['x'].value
        y = point.attributes['y'].value
        coordinate.append([int(x), int(y)])

    poly = np.array(coordinate)

    # (1) Crop the bounding rect
    rect = cv2.boundingRect(poly)
    x, y, w, h = rect
    croped = img[y:y+h, x:x+w].copy()

    # (2) make mask
    poly = poly - poly.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # (3) do bit-op
    roi = cv2.bitwise_and(croped, croped, mask=mask)

    # (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg, bg, mask=mask)
    output = bg + roi

    cv2.imwrite('./train_data/temp/out' + space.attributes['id'].value + '.png', output)
