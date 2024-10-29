import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

# Define the image size
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CHANNELS = 3  # Assuming your images are RGB

DATA_TRAIN = "../dataset/EEG_Spectrograms"
FILE_CONFIG_TRAIN = "../dataset/train.csv"

# Define the number of classes
NUM_CLASSES = 6  # Replace with your actual number of classes

def load_data_frame(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['eeg_id'], keep='first')

    dicionario_paciente = df.set_index('eeg_id').to_dict(orient='index')
    dicionario = df.set_index('eeg_id').to_dict(orient='index')
    print(df.head())
    return df, dicionario_paciente, dicionario

def get_label(id, df):
  v = df[id]['expert_consensus']
  return v

def get_patient(id, df):
  v = df[id]['patient_id']
  return v

def load_data(path_images, image_size, num_classes, dicionarios):
    """Loads images and labels from NPY files."""
    images = []
    labels = []
    groups = []
    df = dicionarios["df"]

    index = 0
    for filename in os.listdir(path_images):

        if filename.endswith(".npy"):
            nome_sem_extensao = os.path.splitext(filename)[0]
            nome_sem_extensao = int(nome_sem_extensao)
            matches = df[df["eeg_id"] == nome_sem_extensao]

            if len(matches) > 0:
                v = matches.values[0]  # Get the first match
            else:
                print(f"No rows found for eeg_id: {nome_sem_extensao}")
                continue

            image_path = os.path.join(path_images, filename)
            image = np.load(image_path)

            # for canal in range(image.shape[2]):
            #
            #     new_image = image[:, :, canal]
            #
            #     new_image = cv2.resize(new_image, image_size)
            #     new_image = np.expand_dims(new_image, axis=-1)  # Adiciona um novo eixo para o canal de cores
            #     new_image = np.repeat(new_image, 3, axis=-1)
            #     images.append(new_image)
            #
            #     label = get_label(nome_sem_extensao, dicionarios["dicionario"])
            #     group = get_patient(nome_sem_extensao, dicionarios["dicionario_paciente"])
            #
            #     labels.append(label)
            #     groups.append(group)


            new_image = np.concatenate((image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]), axis=1)

            new_image = cv2.resize(new_image, image_size)
            new_image = np.expand_dims(new_image, axis=-1)  # Adiciona um novo eixo para o canal de cores
            new_image = np.repeat(new_image, 3, axis=-1)
            images.append(new_image)

            label = get_label(nome_sem_extensao, dicionarios["dicionario"])
            group = get_patient(nome_sem_extensao, dicionarios["dicionario_paciente"])

            labels.append(label)
            groups.append(group)
            index += 1
            if index % 100 == 0:
                print(f">>>> load {index} images")
                if index == 500:
                    break

    print(f"finish load {index} images")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    images = np.array(images)
    images = images.astype('float32') / 255.0

    return images, labels, groups

def create_classifier(image_size, num_classes):

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # Cria o modelo
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compila o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model




if __name__ == "__main__":
    df, dicionario_paciente, dicionario = load_data_frame(FILE_CONFIG_TRAIN)

    dicionarios = {
        "df": df,
        "dicionario_paciente": dicionario_paciente,
        "dicionario": dicionario
    }

    images, labels, groups = load_data(DATA_TRAIN, (IMG_WIDTH, IMG_HEIGHT), NUM_CLASSES, dicionarios)
    n_folds = 5
    print(images.shape)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ... seu código anterior
    results = []

    for fold, (train_index, val_index) in enumerate(sgkf.split(images, labels, groups)):
        print(f"Fold {fold+1}")

        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        model = create_classifier((IMG_WIDTH, IMG_HEIGHT), NUM_CLASSES)  # NUM_CLASSES = 6

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        scory = model.evaluate(X_val, y_val)
        results.append(scory[1])
    average_accuracy = sum(results) / len(results)
    print(f"Acurácia média de cross-validation: {average_accuracy}")
