"""
Driver code for running the convolutional neural network.
Authors: Gareth Nicholas + Emile Givental
Date: December 8th, 2019
"""
# Python imports
import matplotlib.pyplot as plt
import tensorflow as tf

# Our imports
import preprocess
from convnet import *
import tune_models

def main(verbose = False):
    """Main driver code.

    Parameters
    ----------
    verbose : Boolean
        Print and generate debugging information.

    Returns
    -------
    type
        Description of returned object.

    """
    train_data, train_labels = preprocess.load_training_data()
    test_data, test_labels = preprocess.load_testing_data()

    if verbose:
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

    # Validation dataset of size 2000
    val_dset = train_dset.take(2000)
    train_dset = train_dset.skip(2000)

    train_dset = train_dset.shuffle(train_data.shape[0]).batch(64)
    val_dset = val_dset.batch(64)
    test_dset = test_dset.batch(64)

    # Instantiate the model
    model = convnet().model
    tune_models.run_training(model, train_dset, val_dset)
    confusion_matrix = tune_models.run_testing(model, test_dset)

    # Printing the confusion matrix
    num_labels = 43
    row_string = "{:4d}" * num_labels
    # Use some Python unpacking magic to format into the row_string
    labels = row_string.format(*range(num_labels))

    print("\n" + "  " * num_labels + "prediction")
    print("  " + labels)
    print("  " + "____" * num_labels)
    for label in range(num_labels):
        print(str(label) + "|" + row_string.format(*confusion_matrix[label]))

if __name__ == "__main__":
    main(True)
