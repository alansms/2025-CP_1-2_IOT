# Classificador de Cartuchos de Impressora - SPRINT 1

Este repositório contém o projeto desenvolvido durante a SPRINT 1 de 2025, focado na criação de um classificador de cartuchos de impressora utilizando técnicas de visão computacional e aprendizado de máquina.

## Estrutura do Projeto

- **coleta_duckduckgo.py**: Script para coletar imagens de cartuchos de impressora utilizando o mecanismo de busca DuckDuckGo
- **treinamento_modelo.py**: Script para pré-processar as imagens e treinar um modelo de classificação
- **modelo_classificador_cartuchos.h5**: Modelo treinado salvo no formato H5
- **historico_treinamento.png**: Gráfico mostrando a evolução do treinamento do modelo
- **NOTAS_PROJETO.md**: Documentação detalhada sobre o desenvolvimento e resultados do projeto
- **dataset/**: Pasta contendo as imagens coletadas divididas em duas categorias:
  - **HP_Original/**: Imagens de cartuchos HP originais
  - **Outros/**: Imagens de cartuchos de outras marcas (Canon, Epson)

## Objetivos

O objetivo principal deste projeto é desenvolver um classificador capaz de distinguir entre cartuchos HP originais e cartuchos de outras marcas com alta acurácia.

## Resultados Alcançados

- Coleta automatizada de imagens via DuckDuckGo
- Implementação de um modelo CNN com 3 camadas convolucionais
- Aplicação de técnicas de data augmentation para melhorar a generalização
- Acurácia de 90% no conjunto de validação

## Próximos Passos

O desenvolvimento continuará na SPRINT 2, onde serão implementadas melhorias no modelo e a integração com sistemas IoT para classificação em tempo real.

## Requisitos

Para executar os scripts deste projeto, são necessárias as seguintes bibliotecas:
- Python 3.11+
- TensorFlow
- scikit-learn
- matplotlib
- duckduckgo_search
- Pillow (PIL)
- requests

Você pode instalar todas as dependências utilizando:
```
pip install tensorflow scikit-learn matplotlib duckduckgo-search Pillow requests
```

## Como Usar

1. Para coletar novas imagens:
```
python coleta_duckduckgo.py
```

2. Para treinar o modelo:
```
python treinamento_modelo.py
```
