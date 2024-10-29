import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Carregando o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizando os dados de entrada
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convertendo os rótulos para one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Definindo o modelo
model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compilando o modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Treinando o modelo
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Avaliando o modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Fazendo previsões
predictions = model.predict(x_test)

# Mostrando algumas previsões
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Predição: {np.argmax(predictions[i])}")
    plt.axis("off")
plt.show()