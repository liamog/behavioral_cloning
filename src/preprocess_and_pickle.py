import csv
import cv2
import numpy as np
import preprocess as prep
import random
from sklearn.cross_validation import train_test_split
from os import makedirs

CAMERA_ANGLE_CORRECTION = 0.15


def prep_image(folder, source_path):
    source_path = source_path
    filename = source_path.split('/')[-1]
    image_path = folder + '/IMG/' + filename

    return prep.preprocess_image(cv2.imread(image_path))


def process_files_in_folder(folder):
    '''
    folder to search for files
    Expect folder to have form
    driving_log.csv
    IMG/<files...>
    returns a tuple of images and measurements
    '''

    lines = []
    with open(folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if not line:
                continue
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        if (len(line) == 8):
            if int(line[7]) == 0:
                # skip this file.
                continue

        measurement = float(line[3])

        # center image
        measurements.append(measurement)
        center_image =prep_image(folder, line[0])
        images.append(center_image)

        # center image flipped horizontally.
        center_image_mirror = cv2.flip(center_image, 1)
        measurements.append(measurement * -1.0)
        images.append(center_image_mirror)

        # left image
        left_image = prep_image(folder, line[1])
        measurements.append(measurement + CAMERA_ANGLE_CORRECTION)
        images.append(left_image)
        # left image flipped
        measurements.append((measurement + CAMERA_ANGLE_CORRECTION) * -1.0)
        images.append(cv2.flip(left_image, 1))

        # right image
        right_image = prep_image(folder, line[2])
        measurements.append(measurement - CAMERA_ANGLE_CORRECTION)
        images.append(right_image)
        
        # right image mirrored
        measurements.append((measurement - CAMERA_ANGLE_CORRECTION) * -1.0)
        images.append(cv2.flip(right_image, 1))

    print("total = {}, skipped = {}, included(+augs)={}".format(len(
        lines), len(lines) - len(images) / 4, len(images)))
    return images, measurements


folders = ["bridge",
           "cc_lap_with_mouse",
           "focused_center_on_turns",
           "gentle_swerving",
           "lap2_with_mouse",
           "lap_with_mouse",
           "right_turn",
           ]

for folder in folders:

    print ('processing ' + folder)
    images, measurements = process_files_in_folder('../data/raw/' + folder)

    print ('finished {}, {}'.format(folder, len(images)))
    x = np.array(images)
    y = np.array(measurements)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3)

    makedirs("../data/proccessed_and_pickled", exist_ok=True)
    np.savez(
        '../data/proccessed_and_pickled/' + folder + '_train.npz',
        x=x_train,
        y=y_train)
    np.savez(
        '../data/proccessed_and_pickled/' + folder + '_valid.npz',
        x=x_valid,
        y=y_valid)
