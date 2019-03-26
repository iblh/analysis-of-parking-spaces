# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-d", "--dataset", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the image
imagePaths = sorted(list(paths.list_images(args["dataset"])))

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# initialize the data and labels
print("[INFO] loading images...")

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    (empty, occupied) = model.predict(image)[0]

    # build the label
    label = "occupied" if occupied > empty else "empty"
    proba = occupied if occupied > empty else empty
    label = "{}: {:.2f}%".format(label, proba * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=200)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow(imagePath, output)

cv2.waitKey(10000) & 0xFF
