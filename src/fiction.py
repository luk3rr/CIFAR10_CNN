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

STRIDE_VERTICAL = 1
STRIDE_HORIZONTAL = 1
STRIDES = (STRIDE_VERTICAL, STRIDE_HORIZONTAL)

PADDING = 0

NUM_IMAGES = 1000
NUM_LABELS = 5

X = 1  # Número de nós na primeira camada FC
Y = 1  # Número de nós na segunda camada FC

PLOTS_DIR = "plots/"


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


def build_model():
    """
    @brief:
        Definição do modelo de extração de features + classificação

    @return:
        O modelo
    """
    model = Sequential()

    # Camadas de extração das features
    model.add(
        Conv2D(
            1,
            KERNEL_SHAPE,
            strides=STRIDES,
            activation="relu",
            input_shape=INPUT_SHAPE,
        )
    )
    model.add(Conv2D(1, KERNEL_SHAPE, strides=STRIDES, activation="relu"))

    # Camadas fully connected (FC)
    model.add(Flatten())
    model.add(Dense(X, activation="relu"))
    model.add(Dense(Y, activation="linear"))

    # Saída com ativação softmax para classificação
    model.add(Dense(NUM_LABELS, activation="softmax"))

    return model


def split_model(model):
    """
    @brief:
        Separa o modelo de convolução do modelo de classificação
    @return:
        model_cnn, Rede CNN de extração das features
        model_fc, Rede fully connected de classificação
    """
    input_conv = model.input
    # conv_layer1 = model.layers[0]  # Primeira camada Conv2D
    # conv_layer2 = model.layers[1]  # Segunda camada Conv2D
    flatten = model.layers[2]  # Camada Flatten
    input_fc = model.layers[3].input
    # fc_layer1 = model.layers[4]     # Primeira camada Dense após o Flatten
    output_layer = model.layers[5]  # Camada de saída softmax

    # Crie modelos separados para a parte convolucional e totalmente conectada
    cnn_model = tf.keras.models.Model(inputs=input_conv, outputs=flatten.output)
    fc_model = tf.keras.models.Model(inputs=input_fc, outputs=output_layer.output)

    return cnn_model, fc_model


def predict(complete_model, matrix, debug=False):
    """
    @brief:
        Dado um complete_modelo e uma matriz, realiza a classificação desta matriz de acordo com
        o complete_modelo treinado
    """
    input_matrix = matrix.reshape((1, IMG_ROWS, IMG_COLS, CHANNELS))
    predictions = complete_model.predict(input_matrix)

    if debug:
        # Itera sobre as probabilidades e mostra o número da classe e a probabilidade associada
        for class_number, probability in enumerate(predictions.flatten()):
            print(f"Classe {class_number}: {probability}")

    # Para obter a classe prevista, você pode usar a função argmax
    predicted_class = np.argmax(predictions)

    if debug:
        print(
            f"\n# Classe Prevista pela Rede Fully Connected: {predicted_class}",
            end="\n\n",
        )

    return predicted_class


def predict_fc(model_fc, input_value, debug=False):
    """
    @brief:
        Dada uma rede fully connected e um valor de entrada,
        realiza a previsão usando a rede
    """
    # Converte o valor de entrada para o formato esperado pela rede fully connected
    input_matrix = input_value.reshape((1, -1))

    # Realiza a previsão usando a rede fully connected
    predictions = model_fc.predict(input_matrix)

    if debug:
        # Itera sobre as probabilidades e mostra o número da classe e a probabilidade associada
        for class_number, probability in enumerate(predictions.flatten()):
            print(f"Classe {class_number}: {probability}")

    # Para obter a classe prevista, você pode usar a função argmax
    predicted_class = np.argmax(predictions)

    if debug:
        print(
            f"\n# Classe Prevista pela Rede Fully Connected: {predicted_class}",
            end="\n\n",
        )

    return predicted_class


def extract_weights_biases(model):
    """
    @brief:
        Extraí os filtros e as biases de cada camada do modelo
    """
    weights_list = []
    biases_list = []

    for layer in model.layers:
        # Verifica se a camada possui pesos (é uma camada de convolução ou totalmente conectada)
        if hasattr(layer, "get_weights"):
            layer_weights_biases = layer.get_weights()
            if layer_weights_biases:
                layer_weights, layer_biases = map(np.array, layer_weights_biases)

                # Formata a matriz para que ela seja bidimensional
                layer_weights = layer_weights[:, :, 0, 0]

                weights_list.append(layer_weights)
                biases_list.append(layer_biases)

    return weights_list, biases_list


def apply_weights_biases(
    matrix, weights_list, biases_list, stride=STRIDES, debug=False
):
    """
    @brief:
        Aplica manualmente os pesos e biases extraídos a uma matriz
    """
    current_input = matrix
    stride_vertical, stride_horizontal = stride

    if debug:
        print(f"\n# Matriz de Entrada:")
        print(matrix)

    for weights, biases in zip(weights_list, biases_list):
        if weights is not None and biases is not None:
            if debug:
                print(f"\n# Pesos (Filtros) da Camada:")
                print(weights)
                print(f"\n# Biases da Camada:")
                print(biases)

            # Calcula a altura e largura da matriz resultante após a convolução
            altura_saida = (
                int(
                    (current_input.shape[0] - weights.shape[0] + 2 * PADDING)
                    / stride_vertical
                )
                + 1
            )
            largura_saida = (
                int(
                    (current_input.shape[1] - weights.shape[1] + 2 * PADDING)
                    / stride_horizontal
                )
                + 1
            )

            # Inicializa a matriz resultante com zeros
            resultado = np.zeros((altura_saida, largura_saida))

            # Aplica a convolução considerando o stride
            for i in range(0, altura_saida * stride_vertical, stride_vertical):
                for j in range(
                    0, largura_saida * stride_horizontal, stride_horizontal
                ):
                    regiao = current_input[
                        i : i + weights.shape[0], j : j + weights.shape[1]
                    ]

                    if debug:
                        print(f"\n# Região de Entrada na Multiplicação:")
                        print(regiao)
                        print(
                            f"\n# Resultado Parcial Antes da Multiplicação e Adição do Bias:"
                        )
                        print(resultado)

                    # Performa a convolução na região da matriz
                    resultado[i // stride_vertical, j // stride_horizontal] = (
                        np.sum(regiao * weights) + biases
                    )

                    if debug:
                        print(f"\n# Convolução:")
                        for k in range(regiao.shape[0]):
                            for l in range(regiao.shape[1]):
                                print(f"{regiao[k, l]} * {weights[k, l]} + ", end="")

                        print(f"{biases[0]}")

                    if debug:
                        print(
                            f"\n# Resultado Parcial Após a Multiplicação e Adição do Bias:"
                        )
                        print(resultado)

            # Aplica a função de ativação ReLU
            current_input = np.maximum(0, resultado)

            if debug:
                print(f"\n# Matriz após aplicação do filtro + relu:")
                print(current_input)
        else:
            # Se a camada não tem pesos, apenas passa adiante
            current_input = current_input

    h_rear = h_ahead = 0
    for m in range(0, KERNEL_ROWS):
        print()
        # h_rear +=

    return current_input


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
    complete_model = build_model()

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

    cnn_model, fc_model = split_model(complete_model)

    # Avalie o modelo completo
    test_loss, test_acc = complete_model.evaluate(test_images, test_labels, verbose=2)

    print(f"\n-> Test acc: {test_acc}\n-> Test loss: {test_loss}")

    _, fc_model = split_model(complete_model)

    weights_list, biases_list = extract_weights_biases(cnn_model)

    for _ in range(0, 100):
        matrix = np.random.random((IMG_ROWS, IMG_COLS))
        feature = apply_weights_biases(matrix, weights_list, biases_list, debug=False)
        print("#" * 4, " Manual conv ", "#" * 4)
        k = predict_fc(fc_model, feature, debug=True)

        print("#" * 4, " Auto conv ", "#" * 4)
        j = predict(complete_model, matrix, debug=True)
        if k == j:
            print(f"Correct: k = {k} | j = {j}")
        else:
            print("#" * 70)
            print(f"Error: k = {k} | j = {j}")
            print("#" * 70)


if __name__ == "__main__":
    main()
