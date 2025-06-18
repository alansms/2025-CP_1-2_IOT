#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para treinar um modelo de classificação de cartuchos de impressora
utilizando as imagens coletadas pelo script coleta_duckduckgo.py.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Configurações
TAMANHO_IMAGEM = (150, 150)  # Tamanho para redimensionar as imagens
BATCH_SIZE = 16
EPOCAS = 20
TAXA_APRENDIZADO = 0.001
# Usar caminho absoluto para garantir que o script encontre as pastas corretamente
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DATASET = os.path.join(DIRETORIO_ATUAL, "dataset")
CLASSES = ["HP_Original", "Outros"]  # Classes para classificação

def carregar_dados(diretorio_base):
    """
    Carrega as imagens de todas as classes e retorna arrays com dados e rótulos.
    """
    print(f"🔍 Carregando imagens do diretório: {diretorio_base}")

    imagens = []
    rotulos = []

    # Para cada classe
    for idx, classe in enumerate(CLASSES):
        diretorio_classe = os.path.join(diretorio_base, classe)
        print(f"  📁 Processando classe {classe} ({idx})...")

        # Verifica se o diretório existe
        if not os.path.exists(diretorio_classe):
            print(f"⚠️ Diretório {diretorio_classe} não encontrado!")
            continue

        # Carrega todas as imagens do diretório
        for arquivo in os.listdir(diretorio_classe):
            if arquivo.endswith(('.jpg', '.jpeg', '.png')):
                caminho_arquivo = os.path.join(diretorio_classe, arquivo)
                try:
                    # Abre, redimensiona e converte a imagem para array
                    img = Image.open(caminho_arquivo).convert('RGB')
                    img = img.resize(TAMANHO_IMAGEM)
                    img_array = np.array(img) / 255.0  # Normalização

                    imagens.append(img_array)
                    rotulos.append(idx)
                except Exception as e:
                    print(f"⚠️ Erro ao processar {caminho_arquivo}: {e}")

    return np.array(imagens), np.array(rotulos)

def criar_modelo(input_shape, num_classes):
    """
    Cria um modelo CNN simples para classificação de imagens.
    """
    modelo = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Terceira camada convolucional
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten e camadas densas
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Para reduzir overfitting
        Dense(num_classes, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def plotar_historico(historico):
    """
    Plota gráficos de acurácia e perda do treinamento.
    """
    # Acurácia
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Treino')
    plt.plot(historico.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()

    plt.tight_layout()
    plt.savefig('historico_treinamento.png')
    print(f"✅ Gráfico do histórico salvo como 'historico_treinamento.png'")

def main():
    print("🚀 Iniciando processamento e treinamento do modelo")

    # Carregar dados
    X, y = carregar_dados(DIRETORIO_DATASET)

    if len(X) == 0:
        print("❌ Não foram encontradas imagens para treinar o modelo")
        return

    print(f"✅ Carregadas {len(X)} imagens no total")

    # Converter rótulos para one-hot encoding
    y_categorical = to_categorical(y, num_classes=len(CLASSES))

    # Dividir em conjuntos de treino e validação (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )

    print(f"📊 Conjunto de treino: {X_train.shape[0]} imagens")
    print(f"📊 Conjunto de validação: {X_val.shape[0]} imagens")

    # Data augmentation para melhorar a generalização
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Criar e compilar o modelo
    input_shape = (*TAMANHO_IMAGEM, 3)  # (altura, largura, canais)
    modelo = criar_modelo(input_shape, len(CLASSES))

    print("📋 Resumo do modelo:")
    modelo.summary()

    # Callbacks para melhorar o treinamento
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3)
    ]

    # Treinar o modelo
    print("🏋️ Iniciando treinamento do modelo...")
    historico = modelo.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCAS,
        callbacks=callbacks
    )

    # Avaliar o modelo
    print("📏 Avaliando o modelo no conjunto de validação...")
    loss, accuracy = modelo.evaluate(X_val, y_val)
    print(f"📊 Acurácia de validação: {accuracy:.2%}")

    # Salvar o modelo
    modelo.save('modelo_classificador_cartuchos.h5')
    print("💾 Modelo salvo como 'modelo_classificador_cartuchos.h5'")

    # Plotar histórico de treinamento
    plotar_historico(historico)

    print("✅ Processo de treinamento finalizado!")

if __name__ == "__main__":
    main()
