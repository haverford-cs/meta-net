"""
Utilities for preprocessing the German Traffic Sign Dataset. The data we are
working with contains images of various sizes and can be found at
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset. Preprocessing
methods adapted from https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter6/datasets/gtsrb.py.
Authors: Gareth Nicholas + Emile Givental
Date: December 8th, 2019
"""

import cv2
import csv
import os
import numpy as np

training_path = "Training/Final_Training/Images/"
testing_path = "Testing/Final_Test/Images"

def load_training_data():
    """Read in the training data.

    Returns
    -------
    data
        List of cv2 images.
    labels
        Integer label of each cv2 image.

    """
    data = []
    labels = []
    for dir in os.listdir(training_path):
        one_label_dir = os.path.join(training_path, dir)
        # Make sure it is actually a directory
        if os.path.isdir(one_label_dir):
            info_file_name = "GT-" + dir + ".csv"
            csv_path = os.path.join(one_label_dir, info_file_name)
            csv_file = open(csv_path, "r")
            # Not actually comma separated...
            info_reader = csv.reader(csv_file, delimiter = ";")
            # Also skip the header
            next(info_reader)
            for row in info_reader:
                # Path to one image. Slice off around the boundaries of
                # the sign.
                image = cv2.imread(os.path.join(one_label_dir, row[0]))
                image = image[np.int(row[4]):np.int(row[6]),
                            np.int(row[3]):np.int(row[5]), :]
                label = np.int(row[7])
                data.append(image)
                labels.append(label)
            csv_file.close()
    data = normalize(data)
    labels = np.array(labels)
    return data, labels

def load_testing_data():
    """Read in the testing data.

    Returns
    -------
    data
        List of cv2 images.
    labels
        Integer label of each cv2 image.

    """
    data = []
    labels = []
    csv_name = "GT-final_test.csv"
    csv_path = os.path.join(testing_path, csv_name)
    csv_file = open(csv_path, "r")
    # Again, not actually comma separated...
    info_reader = csv.reader(csv_file, delimiter = ";")
    # Skip the header again
    next(info_reader)
    for row in info_reader:
        image = cv2.imread(os.path.join(testing_path, row[0]))
        image = image[np.int(row[4]):np.int(row[6]),
                    np.int(row[3]):np.int(row[5]), :]
        label = np.int(row[7])
        data.append(image)
        labels.append(label)
    csv_file.close()
    data = normalize(data)
    labels = np.array(labels)
    return data, labels


def normalize(data):
    """Resize images to 32x32 then normalize.

    Parameters
    ----------
    data : list
        List of images.

    Returns
    -------
    Numpy array
        Normalized images.

    """
    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data).astype(np.float32)
    mean_pixel = data.mean(axis = (0, 1, 2), keepdims = True)
    std_pixel = data.std(axis=(0, 1, 2), keepdims = True)
    data = (data - mean_pixel) / std_pixel
    return data
