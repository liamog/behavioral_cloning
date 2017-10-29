import csv
import cv2
import numpy as np
import preprocess as prep

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
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/raw/IMG/' + filename
    image = cv2.imread(current_path)
    crop_img = prep.preprocess_image(image)
    images.append(crop_img)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(np.shape(X_train))

X_train.dump("../data/proccessed_and_pickled/x_train.pickle")
y_train.dump("../data/proccessed_and_pickled/y_train.pickle")
