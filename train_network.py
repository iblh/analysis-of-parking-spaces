from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adam
from network.alexnet import AlexNet
from network.vgg13v import VGG13_V
from network.lenet import LeNet
from network.vgg13 import VGG13
from network.vgg16 import VGG16
from network.lenet import LeNet
from network.vgg7 import VGG7
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import random
import cv2
import os
matplotlib.use('Agg')


# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='path to input dataset')
ap.add_argument('-o', '--output', required=True,
                help='path to output model and plot')
args = vars(ap.parse_args())


# 初始化要训练的 Epochs，初始学习率，
# 和 Batch Size (一次迭代中使用的训练样例的数量)
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
# Height Width Depth
IMAGE_DIMS = (40, 40, 3)

# 初始化数据和标签
print('[INFO] loading images...')
data = []
labels = []

# 获取图像路径并进行伪随机
imagePaths = sorted(list(paths.list_images(args['input'])))
random.seed(42)
random.shuffle(imagePaths)


# 循环输入图像
for imagePath in imagePaths:
    # 加载图像，进行预处理，并将其存储在数据列表中
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # 从图像路径中提取标签并更新标签列表
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == 'occupied' else 0
    labels.append(label)

# 将原始像素强度缩放到 [0,1] 范围
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# 使用 80％ 的数据进行训练，并将剩余的 20％ 用于测试
# 将数据划分为训练和测试分组
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# 由于只有两个类 LabelBinarizer 产生 integer 编码, 而不是 one-hot vector
# 编码将分类从整数转换为向量，以应用到 categorical_cross-entropy 为目标函数的模型中
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# 构建用于数据增强的图像生成器
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')


# 初始化 model
print('[INFO] compiling model...')
model = VGG7.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                     depth=IMAGE_DIMS[2], classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# 训练网络
print('[INFO] training network...')
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# 保存模型
print('[INFO] serializing network...')
model.save(args['output'] + '.model')

# 绘制训练损失率和准确率
plt.style.use('ggplot')
fig, ax = plt.subplots(2, sharex=True)
fig.suptitle('Training Loss and Accuracy on empty/occupied')
N = EPOCHS

ax[0].plot(np.arange(0, N), H.history['loss'], label='train_loss')
ax[0].plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
ax[0].legend(loc='lower left')

ax[1].plot(np.arange(0, N), H.history['acc'], label='train_acc')
ax[1].plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
ax[1].legend(loc='lower left')

plt.savefig(args['output'] + '.png')
