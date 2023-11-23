import tensorflow as tf
from tensorflow.keras import layers, models

# Defina sua arquitetura da CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

# Compile o modelo (substitua 'binary_crossentropy' e 'adam' conforme necessário)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treine o modelo com seus dados de treinamento
# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Agora, extraia os pesos da camada CNN e fully connected
cnn_weights = model.layers[0].get_weights()
fc_weights = model.layers[-1].get_weights()

# Se desejar, você pode definir um novo modelo com as camadas específicas
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), weights=cnn_weights))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Flatten())

fc_model = models.Sequential()
fc_model.add(layers.Dense(128, activation='relu', input_shape=(output_shape_of_cnn,), weights=fc_weights))

# Agora você pode usar cnn_model para extrair as características da CNN e fc_model para a rede fully connected
