from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adam
from network.alexnet import AlexNet
from network.tinyvgg import TinyVGG
from network.vgg16 import VGG_16
from network.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import random
import cv2
import os
matplotlib.use("Agg")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="./train_data/models/plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


# initialize the number of epochs to train for, initial learning rate,
# and batch size (the number of training examples utilised in one iteration)
EPOCHS = 5
INIT_LR = 1e-3
BS = 32
# Height Width Depth
IMAGE_DIMS = (40, 40, 3)

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "occupied" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# 初始化 model
print("[INFO] compiling model...")
model = TinyVGG.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                      depth=IMAGE_DIMS[2], classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
fig, ax = plt.subplots(2, sharex=True)
fig.suptitle("Training Loss and Accuracy on empty/occupied")
N = EPOCHS

ax[0].plot(np.arange(0, N), H.history["loss"], label="train_loss")
ax[0].plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
ax[0].legend(loc="lower left")

ax[1].plot(np.arange(0, N), H.history["acc"], label="train_acc")
ax[1].plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
ax[1].legend(loc="lower left")

plt.savefig(args["plot"])
