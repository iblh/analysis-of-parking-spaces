# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-d", "--dataset", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# 加载图片
imagePaths = sorted(list(paths.list_images(args["dataset"])))

# 加载训练完成的网络模型
print("[INFO] loading network...")
model = load_model(args["model"])

# 循环输入图片
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    orig = image.copy()

    # 预处理图像以进行分类
    image = cv2.resize(image, (40, 40))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # 对输入图像进行分类
    (empty, occupied) = model.predict(image)[0]

    # 创建标签
    label = "occupied" if occupied > empty else "empty"
    proba = occupied if occupied > empty else empty
    label = "{}: {:.2f}%".format(label, proba * 100)

    # 在图像上绘制标签
    output = imutils.resize(orig, width=200)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # 显示输出图像
    cv2.imshow(imagePath, output)

cv2.waitKey(10000) & 0xFF
