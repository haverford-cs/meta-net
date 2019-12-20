"""
Driver code for running the convolutional neural network, also generates
a confusion matrix of the results.
Authors: Gareth Nicholas + Emile Givental
Date: December 8th, 2019
"""
# Python imports
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import plot_model
import seaborn as sn
import pandas as pd

# Our imports
import preprocess
from util import parse_args
from convnet import *
from multi_scale_conv import *
from reduced_convnet import *
from comparison_models import *

from knn_baseline import KNN
import tune_models
import os

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def main(verbose = False, validating = False):
    """Main driver code for the project.


    Parameters
    ----------
    verbose : Boolean
        Print and generate debugging information.
    validating : Boolean
        Should a validation set be used.

    Returns
    -------
    type
        Description of returned object.

    """
    
    train_data, train_labels = preprocess.load_training_data(rotating = \
    opts.rotating, shifting = opts.shifting)
    print(opts.rotating)
    test_data, test_labels = preprocess.load_testing_data()

    if opts.verbose:
        print(train_data.shape) # Should be (39209, 32, 32, 3)
        print(train_labels.shape) # Should be (39209,)
        print(test_data.shape) # Should be (12630, 32, 32, 3)
        print(test_labels.shape) # Should be (12630,)
        print(test_labels)

        # Plot data distribution over labels
        fig, ax = plt.subplots(1, 2, sharey = True)
        ax[0].hist(train_labels, 42, color = "r")
        ax[0].set_title("Training data distribution")
        ax[1].hist(test_labels, 42, color = "g")
        ax[1].set_title("Testing data distribution")
        plt.show()

    # Create tensorflow datasets
    train_dset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_dset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    val_dset = None

    # Validation dataset of size 2000
    if opts.validation:

        val_dset = train_dset.take(2000)
        train_dset = train_dset.skip(2000)

    train_dset = train_dset.shuffle(train_data.shape[0]).batch(64)
    test_dset = test_dset.batch(64)
    if opts.validation:
        val_dset = val_dset.batch(64)
    print("before KNN")
    if opts.model == "KNN":
      model = KNN(train_data.reshape(-1, 32*32*3), train_labels)
      acc = model.predict(test_data.reshape(-1, 32*32*3), test_labels)
    else:
      model = eval( f"{opts.model}().model")
      model.summary()
      tune_models.run_training(model, train_dset, opts.validation, val_dset)
      confusion_matrix, acc = tune_models.run_testing(model, test_dset, test_labels)

      # Printing the confusion matrix
      num_labels = 43
      row_string = "{:4d}" * num_labels
      # Use some Python unpacking magic to format into the row_string
      labels = row_string.format(*range(num_labels))
      #plot_confusion_matrix_from_np(confusion_matrix)
      print("\n" + "  " * num_labels + "prediction")
      print("  " + labels)
      print("  " + "____" * num_labels)
      for label in range(num_labels):
          print(str(label) + "|" + row_string.format(*confusion_matrix[label]))
  
  
    if opts.saving:
      f = open("results.txt", "a+")
      f.write(f"The model {opts.model} got an accuracy of {acc} with rotating set to {opts.rotating} and shifting set to {opts.shifting}")
    return acc

    # Also display the matrix in a heatmap!
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(43)],
                      columns = [i for i in range(43)])
    plt.figure(figsize = (10,7))
    # Normalize across rows, sum then divide
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    sn.heatmap(df_cm, annot=False)
    plt.show()

if __name__ == "__main__":
    opts = parse_args()
    main(opts)
