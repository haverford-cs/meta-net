"""
Utilities for preprocessing the German Traffic Sign Dataset. The data we are
working with contains images of various sizes and can be found at
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
Authors: Gareth Nicholas + Emile Givental
Date: December 8th, 2019
"""

import cv2
import csv
import os
import numpy as np



training_path = "Training/Final_Training/Images/"
testing_path = "Testing/Final_Test/"

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
                image = cv2.imread(os.path.join(one_label_dir,row[0]))
                image = image[np.int(row[4]):np.int(row[6]),
                            np.int(row[3]):np.int(row[5]), :]
                label = np.int(row[7])
                data.append(image)
                labels.append(label)
            info_reader.close()
    return data, labels

def normalize(img_data):



if __name__ == "__main__":
    load_training_data()
