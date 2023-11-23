#!/usr/bin/env python3

"""
Filename: cnn.py
Created on: Sep 19, 2023
Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

Description: This script utilizes TensorFlow to train a Convolutional Neural Network (CNN) for image recognition using the CIFAR-10 dataset.
If the 'datasets/cifar-10-batches-py' folder is not found, the script automatically downloads the dataset.
To run this script, follow these steps:
1. Navigate to the 'src' folder.
2. Install the required dependencies by running:
   $ pip install -r requirements.txt
3. Execute the script:
   $ python3 cnn.py
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import pickle
import tarfile
import requests

IMG_ROWS = 32
IMG_COLS = 32
CHANNELS = 3

DATASETS_PATH = '../datasets/'
PLOTS_DIR = '../plots/'
CIFAR_BATCHES_DIR = DATASETS_PATH +  'cifar-10-batches-py'
CIFAR10_URL_PYTHON = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_TAR_FILENAME = 'cifar-10-python.tar.gz'
CLASSES = ('plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def DownloadCifar10():
    """
    # Brief
        Downlaod CIFAR10 dataset
    """
    print("Downloading the CIFAR10 dataset...")

    # Downlaod CIFAR10 dataset as tar file
    response = requests.get(CIFAR10_URL_PYTHON, stream=True)
    if response.status_code == 200:
        with open(DATASETS_PATH + CIFAR10_TAR_FILENAME, 'wb') as f:
            f.write(response.raw.read())

        with tarfile.open(DATASETS_PATH + CIFAR10_TAR_FILENAME, 'r:gz') as tar:
            tar.extractall(path=DATASETS_PATH)

        os.remove(DATASETS_PATH + CIFAR10_TAR_FILENAME)
        print("Download completed")

    else:
        print("An error occurred during the download. Exit the program.")
        exit()

def LoadBatch(filename):
    """
    # Brief
        Load single batch of CIFAR

    # Returns
        A tuple (data, labels)
    """

    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')

        data = datadict['data']
        labels = datadict['labels']

        # Array of uint8.
        # Each row of the array stores a 32x32 colour image.
        # The first 1024 entries contain the red channel values, the next 1024 the green, and
        # the final 1024 the blue.
        data = data.reshape(10000, 3072) # new array with 10000 rows and 3072 columns

        # a list of 10000 numbers in the range 0-9. The number at index i indicates the label
        # of the i-th image in the array data
        labels = np.array(labels)

        return data, labels

def LoadData(path, num_training_images=50000, num_test_images=10000):
    """
    # Brief
        Load all of cifar

    # Returns
        Tuple of numpy arrays: (train_images, train_labels), (test_images, test_labels)
    """
    data = []
    labels = []

    # Load all batch files
    for i in range(1, 6):
        print(os.path.join(path, 'data_batch_%d' % (i, )))
        newData, newLabel = LoadBatch(os.path.join(path, 'data_batch_%d' % (i, )))
        data.append(newData) # add data arrays to the list
        labels.append(newLabel) # add label arrays to the list
        del newData, newLabel

    dataConcatenatedArray = np.concatenate(data)
    labelsConcatenatedArray = np.concatenate(labels)
    dataTestConcatenatedArray, labelsTestConcatenatedArray = LoadBatch(os.path.join(path, 'test_batch'))

    # Convert dada to float32.
    # All the methods or functions in Keras expect the input data to be in default
    # floating datatype (float32)
    dataConcatenatedArray = dataConcatenatedArray.astype('float32')
    dataTestConcatenatedArray = dataTestConcatenatedArray.astype('float32')

    # Normalize pixel values to be between 0 and 1
    # The goal of normalization is to transform features to be on a similar scale.
    # This improves the performance and training stability of the model.
    dataConcatenatedArray /= 255
    dataTestConcatenatedArray /= 255

    dataConcatenatedArray = dataConcatenatedArray.reshape(-1, 32, 32, 3)
    dataTestConcatenatedArray = dataTestConcatenatedArray.reshape(-1, 32, 32, 3)

    return dataConcatenatedArray, labelsConcatenatedArray, dataTestConcatenatedArray, labelsTestConcatenatedArray

def main():
    if not os.path.exists(os.path.join(CIFAR_BATCHES_DIR)):
        DownloadCifar10()

    train_images, train_labels, test_images, test_labels = LoadData(CIFAR_BATCHES_DIR)

    # Define the model as sequential
    model = models.Sequential()

    # CNN Layers
    model.add(layers.InputLayer(input_shape=(IMG_ROWS, IMG_COLS, CHANNELS)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Show the CNN + FC architecture
    model.summary()

    # compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # Iterar sobre todas as camadas e imprimir as matrizes de pesos
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:
            print(f"\nWeights for layer {i} - {layer.name}:")
            print(layer.get_weights()[0])  # Imprime os pesos do kernel
            if len(layer.get_weights()) > 1:
                print(f"\nBiases for layer {i} - {layer.name}:")
                print(layer.get_weights()[1])  # Imprime os pesos do bias

    # Plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    # evaluate the model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)
    plt.savefig(PLOTS_DIR + 'cnn_accuracy.png')

if __name__ == "__main__":
    main()
