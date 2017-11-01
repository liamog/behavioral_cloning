import csv
import cv2
import numpy as np
import preprocess as prep


def crop_image(source_path):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    image_path = '../data/raw/IMG/' + filename

    return prep.preprocess_image(cv2.imread(image_path))


lines = []
with open('../data/raw/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if not line:
            continue
        lines.append(line)

images = []
measurements = []
for line in lines:
    measurement = float(line[3])
    correction = 0.2
    # center image
    measurements.append(measurement)
    center_image = crop_image(line[0])
    images.append(center_image)

    # center image flipped horizontally.
    center_image_mirror = cv2.flip(center_image, 1)
    measurements.append(measurement * -1.0)
    images.append(center_image_mirror)

    # left image
    measurements.append(measurement + correction)
    images.append(crop_image(line[1]))

    # right image
    measurements.append(measurement - correction)
    images.append(crop_image(line[2]))
X_train = np.array(images)
y_train = np.array(measurements)

print(np.shape(X_train))

X_train.dump("../data/proccessed_and_pickled/x_train.pickle")
y_train.dump("../data/proccessed_and_pickled/y_train.pickle")
