import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
from xml.dom import minidom

img_date = '2012-10-12_06_27_31'

# 解析 XML
xmldoc = minidom.parse('./test_images/' + img_date + '.xml')
spacelist = xmldoc.getElementsByTagName('space')

print(len(spacelist))

for space in spacelist:
    # read image as RGB and add alpha (transparency)
    im = Image.open('./test_images/' + img_date + '.jpg').convert("RGBA")

    # convert to numpy (for convenience)
    imArray = np.asarray(im)

    points = space.getElementsByTagName("point")
    
    coordinate = []

    for point in points:

        # 停车位 x&y 坐标
        x = point.attributes['x'].value
        y = point.attributes['y'].value
        coordinate.append((int(x), int(y)))

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(coordinate, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save('./train_data/temp/out' + space.attributes['id'].value + '.png')
