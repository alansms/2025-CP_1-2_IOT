#!/bin/bash

# Script de configuração para o Classificador de Cartuchos HP
# Este script configura um ambiente Conda com todas as dependências necessárias

echo "===== Configurando o ambiente para o Classificador de Cartuchos HP ====="

# Verificar se o Conda está instalado
if ! command -v conda &> /dev/null; then
    echo "Conda não encontrado. Por favor, instale o Miniconda ou Anaconda primeiro."
    echo "Você pode instalar o Miniconda usando: brew install miniconda"
    exit 1
fi

# Nome do ambiente
ENV_NAME="cartucho-env"

# Criar um novo ambiente Conda com Python 3.10 (compatível com TensorFlow)
echo "Criando ambiente Conda '$ENV_NAME' com Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Ativar o ambiente
echo "Ativando o ambiente '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Instalar TensorFlow para macOS (com suporte para Apple Silicon)
echo "Instalando TensorFlow para macOS..."
pip install tensorflow-macos tensorflow-metal

# Instalar outras dependências
echo "Instalando outras dependências (OpenCV, Streamlit, etc.)..."
pip install streamlit opencv-python pillow

echo "===== Configuração concluída! ====="
echo "Para usar o ambiente, execute:"
echo "    conda activate $ENV_NAME"
echo "Para executar o aplicativo, navegue até a pasta do projeto e execute:"
echo "    cd /Users/alansms/PycharmProjects/CHALLENGE_SPRINT-1/2025-SPRINT_2/streamlit_app"
echo "    streamlit run app_real_model.py"
