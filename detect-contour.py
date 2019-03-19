import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from xml.dom import minidom

img_date = '2012-09-11_15_16_58'

img = np.array(Image.open(
    './test_images/' + img_date + '.jpg'), dtype=np.uint8)
fig, ax = plt.subplots(1, figsize=(10, 6))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                    wspace=0, hspace=0)

plt.axis('off')
plt.imshow(img)

# 解析 XML
xmldoc = minidom.parse('./test_images/' + img_date + '.xml')
spacelist = xmldoc.getElementsByTagName('space')

print(len(spacelist))

for space in spacelist:
    # print(space.attributes['id'].value)
    points = space.getElementsByTagName('point')
    coordinate = []

    for point in points:

        # 停车位 x&y 坐标
        x = point.attributes['x'].value
        y = point.attributes['y'].value
        coordinate.append([x, y])

    poly = patches.Polygon(np.array(coordinate), fill=False)
    # Add the patch to the Axes
    ax.add_patch(poly)

plt.show()
