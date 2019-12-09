"""
Functions to tune models on training data then test their performance on
test data.
Authors: Gareth Nicholas + Emile Givental
Date: December 9th, 2019
"""

import tensorflow as tf
import numpy as np

@tf.function
def train_step(model, images, labels, loss_object, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions

@tf.function
def val_step(model, images, labels, loss_object):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    return loss, predictions

def run_training(model, train_dset, validating = False, val_dset = None):
    """Train the model on the training dataset and check its performance
    on the validation dataset.

    Parameters
    ----------
    model : Tensorflow model
        The input model to be trained.
    train_dset : Tensorflow Dataset
        The dataset which the model will be trained on.
    val_dset : Tensorflow Dataset
        The dataset which we will check validation accuracy on.
    validating: Boolean
        Should the validation set be used?

    Returns
    -------
    List of tuples
        Tuples containing the training and validation accuracy for each
        epoch.

    """

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    if validating:
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
            name='val_accuracy')

    # train for 20 epochs (passes over the data)
    for epoch in range(20):
        print(f"Starting epoch {epoch + 1}...")

        for images, labels in train_dset:
            loss, predictions = train_step(model, images, labels,
                loss_object, optimizer)
            train_loss(loss)
            train_accuracy(labels, predictions)

        if not validating:
            template = 'Epoch {}, Loss: {}, Accuracy: {}'
            print(template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100))

        # Then we are using validation set.
        else:
            for images, labels in val_dset:
                loss, predictions = val_step(model, images, labels,
                    loss_object)
                val_loss(loss)
                val_accuracy(labels, predictions)


            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, \
                Val Accuracy: {}'
            print(template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                val_loss.result(),
                                val_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        if validating:
            val_loss.reset_states()
            val_accuracy.reset_states()

def run_testing(model, test_dset):
    """Test the fully trained model on the testing dataset. Generates
    a confusion matrix based on the model's predictions.

    Parameters
    ----------
    model : Tensorflow model
        The trained model
    test_dset : Tensorflow Dataset
        The testing data set

    Returns
    -------
    List of lists
        Generated confusion matrix.

    """

    confusion_matrix = [[0 for i in range(43)]
        for j in range(43)]

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='test_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    for images, labels in test_dset:
        # Does the same thing that a test_step would...
        loss, predictions = val_step(model, images, labels,
            loss_object)
        test_loss(loss)
        test_accuracy(labels, predictions)

        for i, probability_array in enumerate(predictions):
            prediction = np.argmax(probability_array)
            confusion_matrix[labels[i]][prediction] += 1

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss.result(),
                        test_accuracy.result()*100))

    return confusion_matrix
