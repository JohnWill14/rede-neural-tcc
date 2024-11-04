import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50, VGG16
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2
import argparse

# Define the image size
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CHANNELS = 3  # Assuming your images are RGB

DATA_TRAIN = "../dataset/EEG_Spectrograms"
FILE_CONFIG_TRAIN = "../dataset/train.csv"

# Define the number of classes
NUM_CLASSES = 6  # Replace with your actual number of classes

import yaml

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
                if index == 100:
                    break

    print(f"finish load {index} images")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    images = np.array(images)
    images = images.astype('float32') / 255.0

    return images, labels, groups

def create_classifier(config, num_classes):
    """Creates a classifier model based on the specified configuration."""

    model_name = config['model']['name']

    # Create the base model based on the configuration
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=(config['model']['input_shape']['width'],
                                                config['model']['input_shape']['height'],
                                                config['model']['input_shape']['channels']))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                                  input_shape=(config['model']['input_shape']['width'],
                                              config['model']['input_shape']['height'],
                                              config['model']['input_shape']['channels']))
    elif model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False,
                                input_shape=(config['model']['input_shape']['width'],
                                            config['model']['input_shape']['height'],
                                            config['model']['input_shape']['channels']))
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = Flatten()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    print(f"model { config['model']['name']}")

    df, dicionario_paciente, dicionario = load_data_frame(FILE_CONFIG_TRAIN)

    dicionarios = {
        "df": df,
        "dicionario_paciente": dicionario_paciente,
        "dicionario": dicionario
    }

    images, labels, groups = load_data(DATA_TRAIN, (config['model']['input_shape']['width'], config['model']['input_shape']['height']), NUM_CLASSES, dicionarios)
    n_folds = 5
    print(images.shape)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []

    for fold, (train_index, val_index) in enumerate(sgkf.split(images, labels, groups)):
        print(f"Fold {fold+1}")

        X_train = images[train_index]
        y_train = labels[train_index]
        X_val = images[val_index]
        y_val = labels[val_index]

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

        model = create_classifier(config, NUM_CLASSES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


        loss, accuracy = model.evaluate(X_val, y_val)
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        print(f"f1_score type: {type(f1_score)}")

        f1_temp = f1_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
        precision = precision_score(np.argmax(y_val, axis=1), y_pred, average='weighted', zero_division=1)
        recall = recall_score(np.argmax(y_val, axis=1), y_pred, average='weighted')

        results.append([loss, accuracy, f1_temp, precision, recall])

    # Calcule a média das acurácias de todos os folds
    average_accuracy = np.mean(results)
    average_loss = np.mean([result[0] for result in results])
    average_accuracy = np.mean([result[1] for result in results])
    average_f1_score = np.mean([result[2] for result in results])
    average_precision = np.mean([result[3] for result in results])
    average_recall = np.mean([result[4] for result in results])

    # Plot training and validation curves
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='training accuracy', color='red')
    plt.plot(history.history['val_accuracy'], label='validation accuracy', color='blue')
    plt.plot(history.history['loss'], label='training loss', color='red')
    plt.plot(history.history['val_loss'], label='validation loss', color='blue')
    plt.title('Model Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy or Loss')
    plt.legend()
    plt.show()

    print(f"Média - Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}, F1 Score: {average_f1_score:.4f}, Precision: {average_precision:.4f}, Recall: {average_recall:.4f}")