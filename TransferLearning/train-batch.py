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
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2
import argparse
import yaml

# Define o tamanho da imagem
IMG_WIDTH = 128
IMG_HEIGHT = 128
NUM_CHANNELS = 3  # Assumindo que suas imagens são RGB

DATA_TRAIN = "tcc/dataset/dataset/EEG_Spectrograms/"
FILE_CONFIG_TRAIN = "tcc/dataset/hms-harmful-brain-activity-classification/train.csv"

# Define o número de classes
NUM_CLASSES = 6  # Substitua pelo número real de classes

def load_config(config_path):
    """Carrega a configuração de um arquivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data_frame(path):
    """Carrega o DataFrame do arquivo CSV."""
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['eeg_id'], keep='first')

    dicionario_paciente = df.set_index('eeg_id').to_dict(orient='index')
    dicionario = df.set_index('eeg_id').to_dict(orient='index')
    print(df.head())
    return df, dicionario_paciente, dicionario

def get_label(id, df):
    """Obtém o rótulo da imagem."""
    v = df[id]['expert_consensus']
    return v

def get_patient(id, df):
    """Obtém o ID do paciente da imagem."""
    v = df[id]['patient_id']
    return v

def image_generator(path_images, image_size, num_classes, dicionarios, batch_size):
    """Gerador de imagens para carregar em batches."""
    df = dicionarios["df"]

    while True:
        filenames = os.listdir(path_images)
        np.random.shuffle(filenames)  # Embaralha os arquivos para cada época

        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i + batch_size]
            batch_images = []
            batch_labels = []
            batch_groups = []

            for filename in batch_filenames:
                if filename.endswith(".npy"):
                    nome_sem_extensao = os.path.splitext(filename)[0]
                    nome_sem_extensao = int(nome_sem_extensao)
                    matches = df[df["eeg_id"] == nome_sem_extensao]

                    if len(matches) > 0:
                        v = matches.values[0]  # Obtém a primeira correspondência
                    else:
                        print(f"Nenhuma linha encontrada para eeg_id: {nome_sem_extensao}")
                        continue

                    image_path = os.path.join(path_images, filename)
                    image = np.load(image_path)

                    # Processamento da imagem (concatenate e resize)
                    new_image = np.concatenate((image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]), axis=1)
                    new_image = cv2.resize(new_image, image_size)
                    new_image = np.expand_dims(new_image, axis=-1)  # Adiciona um novo eixo para o canal de cores
                    new_image = np.repeat(new_image, 3, axis=-1)

                    # Normalização
                    new_image = new_image.astype('float32') / 255.0

                    # Armazena a imagem, o rótulo e o grupo
                    batch_images.append(new_image)
                    batch_labels.append(get_label(nome_sem_extensao, dicionarios["dicionario"]))
                    batch_groups.append(get_patient(nome_sem_extensao, dicionarios["dicionario_paciente"]))

            # Codifica os rótulos
            label_encoder = LabelEncoder()
            batch_labels = label_encoder.fit_transform(batch_labels)
            batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=num_classes)

            # Converte os dados para arrays NumPy
            batch_images = np.array(batch_images)
            batch_groups = np.array(batch_groups)

            yield batch_images, batch_labels, batch_groups

def create_classifier(config, num_classes):
    """Cria um modelo classificador com base na configuração especificada."""

    model_name = config['model']['name']

    # Cria o modelo base com base na configuração
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
        raise ValueError(f"Nome de modelo inválido: {model_name}")

    # Congela as camadas do modelo base
    for layer in base_model.layers:
        layer.trainable = False

    # Adiciona camadas personalizadas
    x = Flatten()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Cria o modelo
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compila o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":

    config = load_config("tcc/TransferLearning/config.yml")

    print(f"modelo { config['model']['name']}")

    df, dicionario_paciente, dicionario = load_data_frame(FILE_CONFIG_TRAIN)

    dicionarios = {
        "df": df,
        "dicionario_paciente": dicionario_paciente,
        "dicionario": dicionario
    }

    # Define o tamanho do batch
    batch_size = 32

    # Cria o gerador de imagens
    train_generator = image_generator(DATA_TRAIN, (config['model']['input_shape']['width'], config['model']['input_shape']['height']), NUM_CLASSES, dicionarios, batch_size)

    n_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []

    for fold, (train_index, val_index) in enumerate(sgkf.split(df['eeg_id'], df['expert_consensus'], df['patient_id'])):
        print(f"Fold {fold+1}")

        # Obter os IDs dos pacientes para o treino e validação
        train_patient_ids = df['patient_id'].iloc[train_index]
        val_patient_ids = df['patient_id'].iloc[val_index]

        # Criar o modelo classificador
        model = create_classifier(config, NUM_CLASSES)

        # Treinar o modelo
        history = model.fit(
            train_generator,
            epochs=config['model']['epochs'],
            steps_per_epoch=len(train_patient_ids) // batch_size,
            validation_data=image_generator(DATA_TRAIN, (config['model']['input_shape']['width'], config['model']['input_shape']['height']), NUM_CLASSES, dicionarios, batch_size),
            validation_steps=len(val_patient_ids) // batch_size
        )

        # Avaliar o modelo
        loss, accuracy = model.evaluate(
            image_generator(DATA_TRAIN, (config['model']['input_shape']['width'], config['model']['input_shape']['height']), NUM_CLASSES, dicionarios, batch_size),
            steps=len(val_patient_ids) // batch_size
        )

        # Obter as previsões para a validação
        y_pred_proba = model.predict(
            image_generator(DATA_TRAIN, (config['model']['input_shape']['width'], config['model']['input_shape']['height']), NUM_CLASSES, dicionarios, batch_size),
            steps=len(val_patient_ids) // batch_size
        )
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calcular as métricas
        print(f"f1_score type: {type(f1_score)}")

        f1_temp = f1_score(np.argmax(df['expert_consensus'].iloc[val_index].values, axis=0), y_pred, average='weighted')
        precision = precision_score(np.argmax(df['expert_consensus'].iloc[val_index].values, axis=0), y_pred, average='weighted', zero_division=1)
        recall = recall_score(np.argmax(df['expert_consensus'].iloc[val_index].values, axis=0), y_pred, average='weighted')

        results.append([loss, accuracy, f1_temp, precision, recall])

    # Calcular as médias das métricas
    average_loss = np.mean([result[0] for result in results])
    average_accuracy = np.mean([result[1] for result in results])
    average_f1_score = np.mean([result[2] for result in results])
    average_precision = np.mean([result[3] for result in results])
    average_recall = np.mean([result[4] for result in results])

    # Plotar as curvas de treino e validação
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Acurácia de Treino', color='red')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação', color='blue')
    plt.plot(history.history['loss'], label='Perda de Treino', color='red')
    plt.plot(history.history['val_loss'], label='Perda de Validação', color='blue')
    plt.title('Acurácia e Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia ou Perda')
    plt.legend()
    plt.show()

    print(f"Média - Perda: {average_loss:.4f}, Acurácia: {average_accuracy:.4f}, F1 Score: {average_f1_score:.4f}, Precisão: {average_precision:.4f}, Revocação: {average_recall:.4f}")