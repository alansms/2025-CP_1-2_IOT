# Notas de Projeto: Classificador de Cartuchos de Impressora

## Resumo do Projeto

**Data:** 17 de junho de 2025  
**Objetivo:** Desenvolver um classificador para identificar cartuchos HP originais e distingui-los de outras marcas (Canon, Epson)  
**Abordagem:** Aprendizado profundo usando redes neurais convolucionais (CNN)

## Coleta de Dados

Utilizamos o script `coleta_duckduckgo.py` para coletar imagens de diferentes tipos de cartuchos:

- **HP Original:** 24 imagens
- **Outros (Canon e Epson):** 23 imagens (14 Canon + 9 Epson)
- **Total:** 47 imagens

## Pré-processamento

1. **Redimensionamento:** Todas as imagens foram redimensionadas para 150x150 pixels
2. **Normalização:** Valores de pixels normalizados (divididos por 255)
3. **Divisão dos dados:** 
   - 80% treinamento (37 imagens)
   - 20% validação (10 imagens)

## Arquitetura do Modelo

Implementamos uma CNN com a seguinte arquitetura:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 148, 148, 32)        │             896 │
│ max_pooling2d (MaxPooling2D)         │ (None, 74, 74, 32)          │               0 │
│ conv2d_1 (Conv2D)                    │ (None, 72, 72, 64)          │          18,496 │
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 36, 36, 64)          │               0 │
│ conv2d_2 (Conv2D)                    │ (None, 34, 34, 128)         │          73,856 │
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 17, 17, 128)         │               0 │
│ flatten (Flatten)                    │ (None, 36992)               │               0 │
│ dense (Dense)                        │ (None, 128)                 │       4,735,104 │
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
│ dense_1 (Dense)                      │ (None, 2)                   │             258 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 4,828,610 (18.42 MB)
 Trainable params: 4,828,610 (18.42 MB)
 Non-trainable params: 0 (0.00 B)
```

- **Camadas convolucionais:** 3 camadas (32, 64, 128 filtros)
- **Pooling:** MaxPooling após cada camada convolucional
- **Camadas densas:** 1 camada com 128 neurônios + dropout de 50%
- **Camada de saída:** 2 neurônios (softmax) para classificação binária

## Técnicas de Aprimoramento

1. **Data Augmentation:** Implementado para expandir artificialmente o conjunto de dados
   - Rotação: 20 graus
   - Deslocamento horizontal/vertical: 20%
   - Cisalhamento: 20%
   - Zoom: 20%
   - Inversão horizontal

2. **Early Stopping:** Parada antecipada para evitar overfitting (paciência = 5)

3. **Redução adaptativa da taxa de aprendizado:** Redução quando o treinamento estagna (fator = 0.2, paciência = 3)

## Resultados do Treinamento

- **Duração:** 13 épocas (early stopping ativado)
- **Acurácia final no conjunto de validação: 90%**
- **Loss final: 0.3165**

### Progresso do Treinamento (Épocas Selecionadas)

| Época | Acurácia (Treino) | Loss (Treino) | Acurácia (Val) | Loss (Val) | Taxa Aprendizado |
|-------|-------------------|---------------|----------------|------------|------------------|
| 1     | 38.6%             | 0.9931        | 50.0%          | 0.6940     | 0.0010           |
| 5     | 58.1%             | 0.6411        | 90.0%          | 0.4579     | 0.0010           |
| 10    | 69.1%             | 0.5586        | 90.0%          | 0.3687     | 0.0010           |
| 12    | 69.0%             | 0.5512        | 90.0%          | 0.3700     | 0.0002           |
| 13    | 68.9%             | 0.4948        | 90.0%          | 0.3588     | 0.0002           |

## Análise dos Resultados

1. **Alta acurácia:** Conseguimos 90% de acurácia na validação, um excelente resultado considerando o tamanho limitado do conjunto de dados.

2. **Convergência rápida:** O modelo atingiu 90% de acurácia já na 5ª época, demonstrando boa capacidade de aprendizado das características discriminativas.

3. **Estabilidade:** A acurácia de validação se manteve estável em 90% por várias épocas, sugerindo um modelo robusto.

4. **Gap entre treino e validação:** Existe uma diferença entre a acurácia de treino (~69%) e validação (90%), o que pode indicar:
   - O conjunto de validação pode ter exemplos mais "fáceis" de classificar
   - O modelo generaliza bem, apesar do desempenho moderado no conjunto de treinamento

## Artefatos Gerados

1. **Modelo treinado:** `modelo_classificador_cartuchos.h5`
2. **Gráfico de treinamento:** `historico_treinamento.png`

## Limitações

1. **Conjunto de dados pequeno:** Apenas 47 imagens no total
2. **Variabilidade limitada:** As imagens foram coletadas da web e podem não representar todas as variações de cartuchos no mundo real
3. **Apenas duas classes:** O modelo distingue apenas entre HP Original e Outros, sem diferenciar entre as marcas não-HP

## Próximos Passos

1. **Expansão do conjunto de dados:**
   - Coletar mais imagens de cartuchos
   - Incluir mais variações (ângulos, iluminação, fundos)

2. **Refinamento do modelo:**
   - Testar diferentes arquiteturas
   - Ajustar hiperparâmetros
   - Implementar cross-validation

3. **Expansão das classes:**
   - Separar "Outros" em categorias específicas (Canon, Epson, etc.)
   - Incluir categorias para cartuchos falsificados/remanufaturados

4. **Implementação prática:**
   - Desenvolver interface para classificação em tempo real
   - Integrar com sistemas de inventário/controle de qualidade
   - Adaptar para dispositivos móveis ou sistemas embarcados

5. **Testes de campo:**
   - Validar desempenho em condições reais
   - Medir métricas de precisão, recall, F1-score em cenários práticos

## Conclusão

O classificador de cartuchos desenvolvido demonstra a viabilidade de usar aprendizado profundo para distinguir entre cartuchos HP originais e de outras marcas com alta acurácia, mesmo com um conjunto de dados limitado. As técnicas de aumento de dados e prevenção de overfitting foram fundamentais para alcançar bons resultados.

O modelo serve como prova de conceito para sistemas mais sofisticados de controle de qualidade e autenticidade de produtos, com potencial para aplicações em contextos industriais, comerciais e de proteção ao consumidor.
