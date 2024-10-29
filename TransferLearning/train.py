import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import tensorflow_hub as hub
from tensorflow.keras import layers

FILE_TRAIN = '../dataset/train.csv'

def get_dataset_data_frame(show_logs=False):
    caminho_arquivo_file_train = FILE_TRAIN
    df = pd.read_csv(caminho_arquivo_file_train)

    colunas_selecionadas = ['eeg_id', 'expert_consensus']
    df_selecionado = df[colunas_selecionadas]

    df = df_selecionado.drop_duplicates(subset=['eeg_id'], keep='first')

    if show_logs:
        print(df.head())
        print("numero de espectrogramas: ",df.shape[0])
    return df

def get_labels_from_data_frame(df, show_logs=False):
    df_labels = df['expert_consensus']
    df_labels = df_labels.drop_duplicates()

    labels = df_labels.tolist()
    if show_logs:
        print(df_labels.head())
        print(labels)
    return labels


def sets_train(df, n_folds=5):
    X = df.drop('expert_consensus', axis=1)
    y = df['expert_consensus']
    groups = df['patient_id']

    sgkf = StratifiedGroupKFold(n_splits=n_folds)

    X_train_full = []
    y_train_full = []
    X_test = []
    y_test = []

    # Itere sobre os folds
    for train_index, test_index in sgkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        X_train_full.append(X_train)
        y_train_full.append(y_train)

        X_test.append(X_val)
        y_test.append(y_val)

    X_train_full = pd.concat(X_train_full)
    y_train_full = pd.concat(y_train_full)

    X_test = pd.concat(X_test)
    y_test = pd.concat(y_test)

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)
    return X_train_full, y_train_full, X_test, y_test


def get_base_model(name, image_size):
    if name == "imagenet":
        keras_hub = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
    else:
        return None
    return hub.KerasLayer(keras_hub,trainable=False, input_shape=(image_size, image_size, 3))

def generate_model(name_base, image_size):
    base_model = get_base_model(name_base, image_size)
    model = keras.Sequential(
        [
            base_model,
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

def train_model(model, x_train, y_train):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=15, batch_size=32)

def model_evaluete(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


def save_model(model, name):
    model.save(f"../models/{name}.h5")


if __name__ == '__main__':
    print("----------------------------")
    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
    print("----------------------------")

    df = get_dataset_data_frame(show_logs=True)
    labels = get_labels_from_data_frame(df, show_logs=True)
