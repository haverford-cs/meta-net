"""
Driver code for running the convolutional neural network.
Authors: Gareth Nicholas + Emile Givental
Date: December 8th, 2019
"""
# Python imports
import matplotlib.pyplot as plt

# Our imports
import preprocess

def main():
    train_data, train_labels = preprocess.load_training_data()
    test_data, test_labels = preprocess.load_testing_data()
    print(len(train_labels)) # Should be 39209
    print(len(test_labels)) # Should be 12630 

    # Uncomment to get label distributions
    fig, ax = plt.subplots(1, 2, sharey = True)
    ax[0].hist(train_labels, 42, color = "r")
    ax[0].set_title("Training data distribution")
    ax[1].hist(test_labels, 42, color = "g")
    ax[1].set_title("Testing data distribution")
    plt.show()



if __name__ == "__main__":
    main()
