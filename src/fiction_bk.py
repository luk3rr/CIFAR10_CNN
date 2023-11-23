#!/usr/bin/env python3

"""
Filename: fiction.py
Created on: Sep 19, 2023
Author: Lucas Araújo <araujolucas@dcc.ufmg.br>

Description: This script utilizes TensorFlow to train a Convolutional Neural Network (CNN)
for image recognition using fiction dataset.

To run this script, follow these steps:
1. Navigate to the 'src' folder.
2. Install the required dependencies by running:
   $ pip install -r requirements.txt
3. Execute the script:
   $ python3 cnn.py
"""

import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split

IMG_ROWS = 3
IMG_COLS = 3
CHANNELS = 1

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

KERNEL_ROWS = 2
KERNEL_COLS = 2

KERNEL_SHAPE = (KERNEL_ROWS, KERNEL_COLS)

NUM_IMAGES = 1000
NUM_LABELS = 20

X = 64  # Número de nós na primeira camada FC
Y = 10  # Número de nós na segunda camada FC

PLOTS_DIR = "../plots/"

TEST_MATRIX = np.random.random((IMG_ROWS, IMG_COLS))


def generate_images(num_images, save_path="../generated_images", save_imgs=False):
    """
    @brief:
        Gera matrizes fictícias para treinar e avaliar o modelo

    @return:
        Conjunto de imagens e de rótulos gerados
    """
    os.makedirs(save_path, exist_ok=True)

    images = np.random.random((num_images, IMG_ROWS, IMG_COLS))

    if save_imgs:
        for i in range(num_images):
            plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.savefig(f"{save_path}/image_{i}.png")
            plt.close()

    # Gera rótulos fictícios para as imagens
    labels = np.random.randint(1, NUM_LABELS, num_images)

    # Retorna os dados gerados
    return {"images": images, "labels": labels}


def build_cnn_model():
    """
    @brief:
        Define o modelo de extração das features

    @return:
        Modelo CNN
    """
    model = Sequential()

    # Primeira camada convolucional
    model.add(Conv2D(1, KERNEL_SHAPE, activation="relu", input_shape=INPUT_SHAPE))

    # Segunda camada convolucional
    model.add(Conv2D(1, KERNEL_SHAPE, activation="relu"))

    # Camada de saída
    model.add(Flatten())
    model.add(Dense(NUM_LABELS, activation="linear"))

    return model


def build_fully_connected_model(cnn_output_shape, X, Y):
    """
    @brief:
        Define o modelo Fully Connected (FC) para classificação das features
        extraídas da CNN

    @param cnn_output_shape:
        Tuple, dimensões da saída da camada Flatten da CNN
    @param X:
        int, número de nós na primeira camada FC
    @param Y:
        int, número de nós na segunda camada FC

    @return:
        Modelo Fully Connected
    """
    model = Sequential()

    # Camada de Flatten (saída da CNN)
    model.add(Flatten(input_shape=cnn_output_shape))

    # Primeira camada FC com ativação ReLU
    model.add(Dense(X, activation="relu"))

    # Segunda camada FC
    # linear para a classificação
    model.add(Dense(Y, activation="linear"))

    return model


def predict(model, matrix):
    """
    @brief:
        Dado um modelo e uma matriz, realiza a classificação desta matriz de acordo com
        o modelo treinado
    """
    input_matrix = matrix.reshape((1, IMG_ROWS, IMG_COLS, CHANNELS))
    predictions = model.predict(input_matrix)

    # Itera sobre as probabilidades e mostra o número da classe e a probabilidade associada
    for class_number, probability in enumerate(predictions.flatten()):
        print(f"Classe {class_number}: {probability}")

    # Para obter a classe prevista, você pode usar a função argmax
    predicted_class = np.argmax(predictions)
    print("Classe prevista:", predicted_class)


def main():
    """
    @brief:
        Função main
    """
    generated_data = generate_images(num_images=NUM_IMAGES)

    # Divide os dados em conjuntos de treinamento e validação
    train_images, test_images, train_labels, test_labels = train_test_split(
        generated_data["images"],
        generated_data["labels"],
        test_size=0.2,
        random_state=42,
    )

    train_labels -= 1
    test_labels -= 1

    # Define the model as sequential
    cnn_model = build_cnn_model()
    cnn_output_shape = cnn_model.output_shape[1:]  # Exclua a primeira dimensão (lote)
    cnn_output = cnn_model.output

    # Construa a segunda parte da rede (Fully Connected) usando a saída da CNN como entrada
    fc_model = build_fully_connected_model(cnn_output_shape, X, Y)

    # Crie o modelo completo
    complete_model = Sequential([cnn_model, fc_model])

    # Visualize a arquitetura do modelo completo
    complete_model.summary()

    # Compila e treina o modelo completo
    complete_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = complete_model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    # Iterar sobre todas as camadas e imprimir as matrizes de pesos
    for i, layer in enumerate(complete_model.layers):
        if len(layer.get_weights()) > 0:
            print(f"\nWeights for layer {i} - {layer.name}:")
            print(layer.get_weights()[0])  # Imprime os pesos do kernel
            if len(layer.get_weights()) > 1:
                print(f"\nBiases for layer {i} - {layer.name}:")
                print(layer.get_weights()[1])  # Imprime os pesos do bias

    # Plot
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")

    # Avalie o modelo completo
    test_loss, test_acc = complete_model.evaluate(
        test_images, test_labels, verbose=2
    )

    print(test_acc)
    plt.savefig(PLOTS_DIR + "fiction_accuracy.png")

    predict(complete_model, TEST_MATRIX)

    matrix = np.random.random((IMG_ROWS, IMG_COLS))

    predict(complete_model, matrix)


if __name__ == "__main__":
    main()
