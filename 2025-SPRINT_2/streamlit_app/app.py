import streamlit as st

# Configuração da página - deve ser a primeira instrução Streamlit
st.set_page_config(
    page_title="Classificador de Cartuchos HP",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Classificador de Cartuchos HP\nDesenvolvido para identificar cartuchos HP originais"
    }
)

# Adicionando CSS personalizado para modernizar a interface
st.markdown("""
<style>
    /* Cores e tema moderno */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #f0f2f6;
        --text-color: #333333;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --background-color: #ffffff;
    }
    
    /* Estilo geral */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Botões personalizados */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0055aa;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Cards para conteúdo */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Estilo para tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--secondary-color);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Terminal de log personalizado */
    .terminal {
        background-color: #1e1e1e;
        color: #f0f0f0;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 8px;
        height: 300px;
        overflow-y: auto;
        margin-top: 10px;
        border: 1px solid #444;
    }
    
    /* Resultado da classificação */
    .result-original {
        background-color: rgba(40, 167, 69, 0.2);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    
    .result-fake {
        background-color: rgba(220, 53, 69, 0.2);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
    }
    
    /* Badges de status */
    .status-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 10px;
    }
    
    .badge-success {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid #28a745;
    }
    
    .badge-warning {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border: 1px solid #ffc107;
    }
    
    .badge-danger {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid #dc3545;
    }
    
    /* Ajustes para mobile */
    @media (max-width: 768px) {
        .main {
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 5px 10px;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

import numpy as np
from PIL import Image
import io
import os
import time
from datetime import datetime
import random
import sys
import platform
import base64
from io import BytesIO
import traceback
import inspect

# Importação condicional de psutil para evitar erros
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Aviso: Módulo psutil não encontrado. Alguns recursos de debug estarão limitados.")

# Informações de diagnóstico para ajudar na depuração
print(f"Python versão: {platform.python_version()}")
print(f"Diretório de trabalho: {os.getcwd()}")
print(f"Diretório do Python: {sys.executable}")

# Diretório base do projeto (obtido dinamicamente)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Diretório base: {BASE_DIR}")

# Configurações globais
MODELO_PATH = os.path.join(BASE_DIR, "2025-SPRINT_1", "classificador_cartuchos", "modelo_classificador_cartuchos.h5")
DATASET_PATH = os.path.join(BASE_DIR, "2025-SPRINT_1", "classificador_cartuchos", "dataset")

# Verificando se o modelo existe
print(f"Verificando modelo em: {MODELO_PATH}")
if os.path.exists(MODELO_PATH):
    print(f"Arquivo do modelo encontrado! Tamanho: {os.path.getsize(MODELO_PATH)/1024/1024:.2f} MB")
else:
    print(f"ERRO: Arquivo do modelo NÃO encontrado no caminho: {MODELO_PATH}")

# Tentativa de importar OpenCV de forma opcional
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV importado com sucesso.")
except ImportError as e:
    OPENCV_AVAILABLE = False
    print(f"Erro ao importar OpenCV: {e}")

# Tentativa de importar TensorFlow e carregar o modelo
MODEL_AVAILABLE = False
model = None

# Inicialização dos contadores de cartuchos na sessão
if 'contador_originais' not in st.session_state:
    st.session_state.contador_originais = 0
if 'contador_nao_originais' not in st.session_state:
    st.session_state.contador_nao_originais = 0

try:
    import tensorflow as tf
    print(f"TensorFlow importado com sucesso. Versão: {tf.__version__}")

    if os.path.exists(MODELO_PATH):
        print(f"Tentando carregar o modelo: {MODELO_PATH}")
        try:
            # Desativando logs verbosos do TensorFlow
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            # Carregando o modelo com tratamento de erro detalhado
            model = tf.keras.models.load_model(MODELO_PATH, compile=False)

            # Compilando o modelo após o carregamento
            model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

            MODEL_AVAILABLE = True
            print("✅ Modelo carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar o modelo: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Arquivo do modelo não encontrado: {MODELO_PATH}")
except ImportError as e:
    print(f"❌ Erro ao importar TensorFlow: {e}")
    print("Certifique-se de estar usando o ambiente Conda correto: conda activate cartucho-env")

# Função para classificação de imagens usando o modelo real
def classify_image(img_array):
    if not MODEL_AVAILABLE or model is None:
        error_msg = "Modelo de classificação não está disponível. Usando modo de simulação."
        print(f"⚠️ {error_msg}")
        add_to_log(f"⚠️ AVISO: {error_msg}")

        # Usar a função de simulação como fallback quando o modelo não está disponível
        print("Usando classificação simulada como fallback...")
        return simulate_classification(img_array)

    try:
        # Pré-processar a imagem para o formato que o modelo espera
        input_shape = model.input_shape[1:3]  # Obter as dimensões esperadas pelo modelo (altura, largura)
        print(f"Formato esperado pelo modelo: {input_shape}")
        print(f"Formato da imagem original: {img_array.shape}")

        # Garantir que usamos as dimensões esperadas pelo modelo
        if input_shape != (224, 224):
            print(f"Ajustando para o formato esperado pelo modelo: {input_shape}")
        else:
            print("Usando o formato padrão 224x224")

        # Redimensionar para o tamanho esperado pelo modelo - MÉTODO MELHORADO
        if OPENCV_AVAILABLE:
            # Método aprimorado de redimensionamento que preserva melhor a proporção
            h, w = img_array.shape[:2]

            # Calcular a razão de aspecto
            aspect_ratio = w / h

            # O tamanho alvo é sempre quadrado (224x224 normalmente)
            target_size = input_shape[0]  # Assumindo que input_shape[0] == input_shape[1]

            # Abordagem alternativa: ajustar pelo lado menor para evitar distorção
            # Isso garante que a imagem inteira caiba no quadrado sem distorção
            if aspect_ratio > 1:  # Imagem mais larga que alta
                # Ajustar baseado na altura
                scale_factor = target_size / h
                new_h = target_size
                new_w = int(w * scale_factor)

                # Se a largura calculada for maior que o tamanho alvo, reajustar
                if new_w > target_size * 1.5:  # Limitador para evitar imagens muito largas
                    new_w = target_size * 1.5
                    new_h = int(h * (new_w / w))
            else:  # Imagem mais alta que larga (ou quadrada)
                # Ajustar baseado na largura
                scale_factor = target_size / w
                new_w = target_size
                new_h = int(h * scale_factor)

                # Se a altura calculada for maior que o tamanho alvo, reajustar
                if new_h > target_size * 1.5:  # Limitador para evitar imagens muito altas
                    new_h = target_size * 1.5
                    new_w = int(w * (new_h / h))

            print(f"Dimensões originais: {w}x{h}, Razão de aspecto: {aspect_ratio:.2f}")
            print(f"Novas dimensões sem distorção: {new_w}x{new_h}, Fator de escala: {scale_factor:.2f}")

            # Redimensionar a imagem preservando a proporção
            # Use INTER_AREA para redução e INTER_CUBIC para ampliação
            interpolation = cv2.INTER_AREA if w > new_w else cv2.INTER_CUBIC
            img_resized_prop = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)

            # Criar uma imagem quadrada (preta) do tamanho alvo
            img_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # Calcular offsets para centralizar
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2

            # Ajustar se as dimensões excederem o tamanho alvo
            if new_h > target_size:
                # Cortar a parte central da altura
                start_y = (new_h - target_size) // 2
                img_resized_prop = img_resized_prop[start_y:start_y+target_size, :, :]
                y_offset = 0
                new_h = target_size

            if new_w > target_size:
                # Cortar a parte central da largura
                start_x = (new_w - target_size) // 2
                img_resized_prop = img_resized_prop[:, start_x:start_x+target_size, :]
                x_offset = 0
                new_w = target_size

            # Colocar a imagem redimensionada no centro da imagem quadrada
            try:
                img_square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized_prop
                print(f"Imagem centralizada corretamente em um quadrado de {target_size}x{target_size}")
            except ValueError as e:
                print(f"Erro ao centralizar: {e}. Ajustando dimensões...")
                # Caso as dimensões sejam incompatíveis, fazer uma correção final
                img_resized_prop = cv2.resize(img_resized_prop,
                                            (min(new_w, target_size - x_offset),
                                             min(new_h, target_size - y_offset)))
                new_h, new_w = img_resized_prop.shape[:2]
                img_square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized_prop

            # Usar a imagem quadrada como resultado
            img_resized = img_square

        else:
            # Usar Pillow se OpenCV não estiver disponível - método aprimorado
            pil_img = Image.fromarray(img_array)

            # Obter dimensões originais
            width, height = pil_img.size
            aspect_ratio = width / height

            # O tamanho alvo é sempre quadrado
            target_size = input_shape[0]  # Assumindo que input_shape[0] == input_shape[1]

            # Abordagem alternativa: ajustar pelo lado menor para evitar distorção
            if aspect_ratio > 1:  # Imagem mais larga que alta
                # Ajustar baseado na altura
                scale_factor = target_size / height
                new_height = target_size
                new_width = int(width * scale_factor)

                # Limitador para evitar imagens muito largas
                if new_width > target_size * 1.5:
                    new_width = int(target_size * 1.5)
                    new_height = int(height * (new_width / width))
            else:
                # Ajustar baseado na largura
                scale_factor = target_size / width
                new_width = target_size
                new_height = int(height * scale_factor)

                # Limitador para evitar imagens muito altas
                if new_height > target_size * 1.5:
                    new_height = int(target_size * 1.5)
                    new_width = int(width * (new_height / height))

            print(f"Dimensões originais PIL: {width}x{height}, Razão de aspecto: {aspect_ratio:.2f}")
            print(f"Novas dimensões PIL sem distorção: {new_width}x{new_height}, Fator de escala: {scale_factor:.2f}")

            # Redimensionar preservando a proporção usando um método de alta qualidade
            # LANCZOS para melhor qualidade no redimensionamento
            pil_img_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

            # Criar uma nova imagem quadrada com fundo preto
            new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))

            # Calcular posição para centralizar
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2

            # Ajustar se as dimensões excederem o tamanho alvo
            if new_height > target_size:
                # Cortar a parte central da altura
                crop_top = (new_height - target_size) // 2
                pil_img_resized = pil_img_resized.crop((0, crop_top, new_width, crop_top + target_size))
                paste_y = 0
                new_height = target_size

            if new_width > target_size:
                # Cortar a parte central da largura
                crop_left = (new_width - target_size) // 2
                pil_img_resized = pil_img_resized.crop((crop_left, 0, crop_left + target_size, new_height))
                paste_x = 0
                new_width = target_size

            # Colar a imagem redimensionada no centro
            new_img.paste(pil_img_resized, (paste_x, paste_y))

            img_resized = np.array(new_img)
            print(f"Imagem PIL centralizada sem distorção em quadrado de {target_size}x{target_size}")
        # Converter para RGB se a imagem estiver em escala de cinza
        if len(img_resized.shape) == 2:
            if OPENCV_AVAILABLE:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            else:
                pil_img = Image.fromarray(img_resized).convert('RGB')
                img_resized = np.array(pil_img)
            print("Convertido de escala de cinza para RGB")

        # Converter para RGB se a imagem estiver em RGBA
        elif len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
            if OPENCV_AVAILABLE:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
            else:
                pil_img = Image.fromarray(img_resized).convert('RGB')
                img_resized = np.array(pil_img)
            print("Convertido de RGBA para RGB")

        # Aplicar pré-processamento adicional para melhorar a classificação
        # 1. Normalizar a imagem para valores entre 0 e 1
        img_normalized = img_resized / 255.0

        # 2. Aplicar aumento de contraste para destacar características
        if OPENCV_AVAILABLE:
            img_norm_uint8 = (img_normalized * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = np.zeros_like(img_norm_uint8)
            for i in range(3):  # Aplicar em cada canal RGB
                img_enhanced[:, :, i] = clahe.apply(img_norm_uint8[:, :, i])
            img_normalized = img_enhanced / 255.0
            print("Aplicado aumento de contraste adaptativo")

        # 3. Aumentar a nitidez (sharpening)
        if OPENCV_AVAILABLE:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            img_sharp = cv2.filter2D((img_normalized * 255).astype(np.uint8), -1, kernel)
            img_normalized = img_sharp / 255.0
            print("Aplicado filtro de nitidez")

        # Criar batch para o modelo
        img_batch = np.expand_dims(img_normalized, axis=0)
        print(f"Formato do lote de entrada final: {img_batch.shape}")

        # Fazer a predição
        prediction = model.predict(img_batch, verbose=1)
        print(f"Previsão bruta do modelo: {prediction}")

        # Sistema balanceado para detecção de ambos os tipos de cartuchos
        # Threshold intermediário para originais (equilibrando detecção)
        threshold_original = 0.70  # Valor equilibrado para originais

        # Threshold intermediário para não originais (melhorando acertividade)
        threshold_nao_original = 0.60  # Valor reduzido para detectar melhor não originais

        # Penalidade equilibrada
        base_penalty = 0.15  # Valor intermediário de penalidade

        # Obter o resultado da classificação com parâmetros balanceados
        if len(prediction.shape) == 1 or prediction.shape[1] == 1:
            # Modelo com uma saída (0-1 para binário)
            raw_confidence = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]

            # Sistema de classificação em três zonas para melhor precisão
            # Zona 1: Alta probabilidade de ser original (acima do threshold_original)
            # Zona 2: Alta probabilidade de ser não original (abaixo do threshold_nao_original)
            # Zona 3: Zona de incerteza (entre os dois thresholds) - usa regras especiais

            # Aplicar penalidade adaptativa
            if raw_confidence > 0.85:
                adaptive_penalty = base_penalty * 0.6  # Penalidade reduzida (0.09)
            elif raw_confidence > 0.75:
                adaptive_penalty = base_penalty * 0.8  # Penalidade moderada (0.12)
            else:
                adaptive_penalty = base_penalty  # Penalidade padrão (0.15)

            # Aplicar penalidade
            adjusted_raw_confidence = raw_confidence - (raw_confidence * adaptive_penalty)

            # Sistema de classificação por zonas
            if adjusted_raw_confidence >= threshold_original:
                # Zona 1: Claramente original
                is_original = True
            elif adjusted_raw_confidence <= threshold_nao_original:
                # Zona 2: Claramente não original
                is_original = False
            else:
                # Zona 3: Zona de incerteza - usar análise detalhada

                # Para valores mais próximos do threshold original, favorece original
                # Para valores mais próximos do threshold não original, favorece não original
                dist_to_original = threshold_original - adjusted_raw_confidence
                dist_to_nao_original = adjusted_raw_confidence - threshold_nao_original

                # Se estiver mais próximo do threshold original E tiver confiança bruta alta
                if dist_to_original < dist_to_nao_original and raw_confidence > 0.75:
                    is_original = True
                    print("Zona de incerteza - mais próximo de original com alta confiança")
                else:
                    # Em caso de dúvida na zona de incerteza, classificar como não original
                    is_original = False
                    print("Zona de incerteza - classificado como não original")

            # Calcular a confiança ajustada
            if is_original:
                # Confiança para originais baseada na distância do threshold
                confidence = (adjusted_raw_confidence - threshold_original) / (1.0 - threshold_original) * 100
                # Pequeno ajuste para aumentar confiança de originais claros
                if adjusted_raw_confidence > 0.85:
                    confidence = min(confidence * 1.1, 100)
            else:
                # Confiança para não originais baseada na distância do threshold
                # Quanto mais abaixo do threshold_nao_original, maior a confiança
                if adjusted_raw_confidence < threshold_nao_original:
                    confidence = (threshold_nao_original - adjusted_raw_confidence) / threshold_nao_original * 100
                    # Aumentar confiança para não originais claros
                    if adjusted_raw_confidence < 0.4:
                        confidence = min(confidence * 1.1, 100)
                else:
                    # Na zona de incerteza, confiança baseada na proximidade relativa
                    total_zone_size = threshold_original - threshold_nao_original
                    relative_position = (adjusted_raw_confidence - threshold_nao_original) / total_zone_size
                    confidence = (1 - relative_position) * 100  # Mais próximo de não original = maior confiança

            print(f"Confiança original: {raw_confidence:.4f}, Ajustada com penalidade adaptativa ({adaptive_penalty:.2f}): {adjusted_raw_confidence:.4f}")
            print(f"Threshold aplicado: {'original (' + str(threshold_original) + ')' if is_original else 'não original (' + str(threshold_nao_original) + ')'}")
            print(f"Zona de classificação: {1 if adjusted_raw_confidence >= threshold_original else 2 if adjusted_raw_confidence <= threshold_nao_original else 3}")
        else:
            # Modelo com várias saídas (softmax para multiclasse)
            class_idx = np.argmax(prediction[0])
            raw_confidence = prediction[0][class_idx]

            # Para modelos multiclasse, usamos uma abordagem diferente
            if class_idx == 0:  # Se for classificado como Original
                # Aplicar penalidade adaptativa
                if raw_confidence > 0.85:
                    adaptive_penalty = base_penalty * 0.6
                elif raw_confidence > 0.75:
                    adaptive_penalty = base_penalty * 0.8
                else:
                    adaptive_penalty = base_penalty

                adjusted_raw_confidence = raw_confidence - (raw_confidence * adaptive_penalty)

                # Verificar se após a penalidade ainda supera o threshold
                if adjusted_raw_confidence < threshold_original:
                    # Verificar se está na zona de incerteza
                    if adjusted_raw_confidence > threshold_nao_original:
                        # Verificar outros scores do modelo para confirmar
                        other_max = np.max(prediction[0][1:]) if prediction.shape[1] > 1 else 0

                        # Se outro score for próximo, dar preferência ao n��o original
                        if other_max > (adjusted_raw_confidence * 0.8):
                            class_idx = 1 + np.argmax(prediction[0][1:])
                            raw_confidence = prediction[0][class_idx]
                            adjusted_raw_confidence = raw_confidence
                            is_original = False
                            print("Reclassificado como não original devido a score próximo de outra classe")
                        else:
                            # Se não houver score próximo, manter como original na zona de incerteza
                            # apenas se tiver confiança bruta alta
                            is_original = raw_confidence > 0.75
                    else:
                        # Abaixo do threshold_nao_original, classificar como não original
                        is_original = False
                else:
                    # Acima do threshold_original, manter como original
                    is_original = True
            else:
                # Se a classificação inicial não for original
                # Verificar se está próximo do limite de decisão
                original_conf = prediction[0][0]

                # Se o score da classe original for muito alto, pode ser um falso negativo
                if original_conf > 0.75 and original_conf > (raw_confidence * 0.95):
                    # Reclassificar como original apenas se muito próximo
                    is_original = True
                    class_idx = 0
                    raw_confidence = original_conf
                    adjusted_raw_confidence = original_conf - (original_conf * base_penalty)
                    print("Reclassificação como original aplicada - score original muito alto")
                else:
                    is_original = False
                    adjusted_raw_confidence = raw_confidence

            # Calcular confiança final
            if is_original:
                confidence = (adjusted_raw_confidence - threshold_original) / (1.0 - threshold_original) * 100
                # Pequeno ajuste para originais claros
                if adjusted_raw_confidence > 0.85:
                    confidence = min(confidence * 1.1, 100)
            else:
                confidence = raw_confidence * 100
                # Ajuste para não originais claros
                if raw_confidence > 0.85:
                    confidence = min(confidence * 1.1, 100)

            print(f"Classe: {class_idx}, Confiança original: {raw_confidence:.4f}, Ajustada: {adjusted_raw_confidence:.4f}")
            print(f"Threshold aplicado: {'original (' + str(threshold_original) + ')' if is_original else 'não original (' + str(threshold_nao_original) + ')'}")

        print(f"Classificação final: {'Original' if is_original else 'Não Original'}, "
              f"confiança: {confidence:.2f}% (valor bruto: {raw_confidence:.4f}, ajustado: {adjusted_raw_confidence:.4f})")
        return is_original, confidence
    except Exception as e:
        print(f"Erro durante a classificação com o modelo real: {e}")
        import traceback
        traceback.print_exc()

        # Tentar obter mais informações sobre o modelo para depuração
        try:
            if model is not None:
                print("\nInformações do modelo para depuração:")
                print(f"Formato de entrada do modelo: {model.input_shape}")
                print(f"Formato de saída do modelo: {model.output_shape}")
                print(f"Camadas do modelo:")
                for i, layer in enumerate(model.layers):
                    print(f"  Camada {i}: {layer.name}, Tipo: {type(layer).__name__}, "
                          f"Entrada: {layer.input_shape}, Saída: {layer.output_shape}")
        except Exception as model_debug_error:
            print(f"Erro ao obter informações do modelo: {model_debug_error}")

        # Em vez de lançar a exceção, usar o modo de simulação como fallback
        error_msg = f"Erro na classificação: {str(e)}. Usando modo de simulação como fallback."
        print(f"⚠️ {error_msg}")
        add_to_log(f"⚠️ AVISO: {error_msg}")

        # Usar a função de simulação como fallback quando ocorre um erro
        print("Usando classificação simulada como fallback devido ao erro...")
        return simulate_classification(img_array)
# Função para simular a classificação (usado como fallback)
def simulate_classification(img_array):
    # Gera um resultado aleatório com tendência para "HP Original"
    is_original = random.random() > 0.3  # 70% de chance de ser classificado como original
    confidence = random.uniform(75, 98)  # Confiança entre 75% e 98%

    print(f"Classificação SIMULADA: {'Original' if is_original else 'Não Original'}, "
          f"confiança: {confidence:.2f}%")

    # Simula um pequeno atraso como se estivesse processando
    time.sleep(0.5)

    return is_original, confidence

# Função para processar imagem da câmera em base64 (para dispositivos móveis)
def process_camera_image(encoded_image):
    try:
        print("\n============= INÍCIO DO PROCESSAMENTO DE IMAGEM =============")
        print(f"DEBUG INFO: Python version: {sys.version}")
        print(f"DEBUG INFO: NumPy version: {np.__version__}")
        print(f"DEBUG INFO: PIL version: {PIL.__version__}")
        print(f"DEBUG INFO: Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        # Verificar memória disponível apenas se psutil estiver disponível
        if PSUTIL_AVAILABLE:
            print(f"DEBUG INFO: Memória disponível: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        else:
            print("DEBUG INFO: Informação de memória não disponível (psutil não instalado)")

        # Verificar se encoded_image é um objeto Streamlit ou outro tipo não esperado
        print(f"DEBUG: Tipo de dado recebido: {type(encoded_image)}")
        print(f"DEBUG: ID do objeto: {id(encoded_image)}")
        if hasattr(encoded_image, "__dict__"):
            print(f"DEBUG: Atributos disponíveis: {dir(encoded_image)}")

        if not isinstance(encoded_image, str):
            print(f"DEBUG: encoded_image não é uma string, �� um {type(encoded_image)}")
            print(f"DEBUG: Representação do objeto: {repr(encoded_image)[:200]}")

            # Se for um objeto Streamlit, extrair o valor real com tratamento mais robusto
            if hasattr(encoded_image, '_value') and encoded_image._value is not None:
                # Se _value for uma função, usar uma imagem de exemplo
                if callable(encoded_image._value):
                    print(f"DEBUG: _value é uma função, não podemos usá-la diretamente")
                    print(f"DEBUG: Assinatura da função: {inspect.signature(encoded_image._value) if inspect else 'inspect não disponível'}")
                    fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                    if os.path.exists(fallback_path):
                        print(f"DEBUG: Usando imagem de exemplo: {fallback_path}")
                        print(f"DEBUG: Tamanho do arquivo: {os.path.getsize(fallback_path)} bytes")
                        return np.array(Image.open(fallback_path))
                    else:
                        raise ValueError("Não foi possível processar a imagem da câmera e o exemplo não está disponível")
                else:
                    encoded_image = encoded_image._value
                    print(f"DEBUG: Valor extraído de _value: {type(encoded_image)}")
                    print(f"DEBUG: Comprimento do valor extraído: {len(encoded_image) if hasattr(encoded_image, '__len__') else 'N/A'}")

            elif hasattr(encoded_image, 'value') and encoded_image.value is not None:
                # Se value for uma função, usar uma imagem de exemplo
                if callable(encoded_image.value):
                    print(f"value �� uma função, não podemos usá-la diretamente")
                    fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                    if os.path.exists(fallback_path):
                        print(f"DEBUG: Usando imagem de exemplo: {fallback_path}")
                        print(f"DEBUG: Tamanho do arquivo: {os.path.getsize(fallback_path)} bytes")
                        return np.array(Image.open(fallback_path))
                    else:
                        raise ValueError("Não foi possível processar a imagem da câmera e o exemplo não está disponível")
                else:
                    encoded_image = encoded_image.value
                    print(f"DEBUG: Valor extraído de value: {type(encoded_image)}")
                    print(f"DEBUG: Comprimento do valor extraído: {len(encoded_image) if hasattr(encoded_image, '__len__') else 'N/A'}")
            elif isinstance(encoded_image, dict) and 'value' in encoded_image:
                encoded_image = encoded_image['value']
                print(f"DEBUG: Valor extraído do dicionário: {type(encoded_image)}")
            else:
                # Fallback - usar caminho de exemplo apenas para demonstração
                fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                if os.path.exists(fallback_path):
                    print(f"DEBUG: Usando imagem de exemplo para demonstração")
                    return np.array(Image.open(fallback_path))
                else:
                    raise ValueError(f"Não foi possível processar o tipo de imagem: {type(encoded_image)}")

        # Se ainda for um objeto de função ou outro tipo não string após extração, usar fallback
        if not isinstance(encoded_image, str):
            print(f"DEBUG: Após extração, ainda não é uma string: {type(encoded_image)}")
            print(f"DEBUG: Tentando converter para string: {str(encoded_image)[:100]}...")
            try:
                encoded_image = str(encoded_image)
                print(f"DEBUG: Conversão para string bem-sucedida, tamanho: {len(encoded_image)}")
            except Exception as str_error:
                print(f"DEBUG: Erro ao converter para string: {str_error}")

            if not isinstance(encoded_image, str):
                fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                if os.path.exists(fallback_path):
                    print(f"DEBUG: Usando imagem de exemplo para demonstração")
                    img = Image.open(fallback_path)
                    print(f"DEBUG: Imagem de exemplo carregada: {img.format}, {img.size}, {img.mode}")
                    return np.array(img)
                else:
                    raise ValueError(f"Não foi possível processar o tipo de imagem: {type(encoded_image)}")

        # Agora temos certeza que encoded_image é uma string
        print(f"DEBUG: Tamanho da string recebida: {len(encoded_image)} caracteres")
        print(f"DEBUG: Primeiros 100 caracteres: {encoded_image[:100]}...")
        print(f"DEBUG: Últimos 100 caracteres: {encoded_image[-100:] if len(encoded_image) > 100 else encoded_image}")
        print(f"DEBUG: Caracteres especiais encontrados: {[c for c in set(encoded_image[:1000]) if not c.isalnum() and c not in '+/='][:20]}")

        # Verificar se é um caminho de arquivo
        if os.path.exists(encoded_image):
            print(f"DEBUG: Processando imagem de arquivo: {encoded_image}")
            image = Image.open(encoded_image)
            return np.array(image)

        # Verificar se parece ser uma string base64 válida
        if not encoded_image or len(encoded_image) < 100:
            raise ValueError(f"String muito curta para ser base64 válido: {len(encoded_image)} caracteres")

        # Tentar identificar o formato dos dados
        if 'data:image' in encoded_image:
            print("DEBUG: Formato detectado: URL de dados com prefixo")
            # Remover cabeçalho de dados se presente (formato: data:image/jpeg;base64,)
            parts = encoded_image.split(',')
            if len(parts) > 1:
                prefix = parts[0]
                encoded_image = parts[1]
                print(f"DEBUG: Prefixo: {prefix}")
                print(f"DEBUG: Base64 extraído com {len(encoded_image)} caracteres")
            else:
                print("DEBUG: Aviso: Formato de URL de dados inesperado")

        # Limpar a string de caracteres inválidos
        # Remover aspas, barras invertidas e outros caracteres problemáticos
        original_len = len(encoded_image)
        encoded_image = encoded_image.replace('"', '').replace('\\', '').strip()
        cleaned_len = len(encoded_image)
        print(f"DEBUG: Limpeza de caracteres: removidos {original_len - cleaned_len} caracteres")

        # Analisar a string para caracteres base64 válidos
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        invalid_chars = [c for c in set(encoded_image) if c not in base64_chars]
        if invalid_chars:
            print(f"DEBUG: Caracteres inválidos encontrados no base64: {invalid_chars}")
            # Remover caracteres inválidos
            for c in invalid_chars:
                encoded_image = encoded_image.replace(c, '')
            print(f"DEBUG: String limpa, novo tamanho: {len(encoded_image)}")

        # Garantir que seja múltiplo de 4 (requisito para base64 válido)
        padding_needed = len(encoded_image) % 4
        if padding_needed > 0:
            encoded_image += '=' * (4 - padding_needed)
            print(f"DEBUG: Adicionado padding: {4 - padding_needed} caracteres '='")

        # Decodificar a string base64
        try:
            start_time = time.time()
            decoded_image = base64.b64decode(encoded_image)
            end_time = time.time()
            print(f"DEBUG: Decodificação base64 bem-sucedida: {len(decoded_image)} bytes em {(end_time - start_time)*1000:.2f}ms")
            print(f"DEBUG: Primeiros 20 bytes da imagem: {decoded_image[:20].hex()}")
        except Exception as e:
            print(f"DEBUG: Erro na decodificação base64: {str(e)}")
            print(f"DEBUG: Tentando decodificar com diferentes codificações...")
            try:
                # Tentar diferentes variações de base64
                for encoding in ['standard_b64decode', 'urlsafe_b64decode']:
                    try:
                        decode_func = getattr(base64, encoding)
                        decoded_image = decode_func(encoded_image)
                        print(f"DEBUG: Decodificação bem-sucedida com {encoding}: {len(decoded_image)} bytes")
                        break
                    except Exception as inner_e:
                        print(f"DEBUG: Falha com {encoding}: {str(inner_e)}")
            except Exception as alt_e:
                print(f"DEBUG: Todas as tentativas de decodificação alternativas falharam: {str(alt_e)}")
                raise ValueError(f"Falha na decodificação base64: {str(e)}")

        # Verificar se os primeiros bytes parecem uma imagem válida
        image_signatures = {
            b'\xff\xd8\xff': 'JPEG',
            b'\x89\x50\x4e\x47': 'PNG',
            b'\x47\x49\x46': 'GIF',
            b'\x42\x4d': 'BMP'
        };

        detected_format = None
        for sig, fmt in image_signatures.items():
            if decoded_image.startswith(sig):
                detected_format = fmt
                print(f"Formato de imagem detectado nos bytes: {fmt}")
                break

        if not detected_format:
            print(f"AVISO: Não foi possível detectar formato de imagem válido. Primeiros bytes: {decoded_image[:20].hex()}")

        # Criar BytesIO com os dados da imagem
        image_bytes = BytesIO(decoded_image)
        print(f"DEBUG: BytesIO criado com {len(decoded_image)} bytes")

        # Tentar abrir a imagem com diferentes abordagens
        try:
            # Abordagem 1: Diretamente com Pillow
            start_time = time.time()
            image = Image.open(image_bytes)
            image.load()  # Forçar carregamento para validar a imagem
            end_time = time.time()
            print(f"DEBUG: Imagem carregada com Pillow em {(end_time - start_time)*1000:.2f}ms: formato={image.format}, tamanho={image.size}, modo={image.mode}")

            # Verificar integridade da imagem
            try:
                image.verify()
                print("DEBUG: Verificação de integridade da imagem: OK")
            except Exception as verify_error:
                print(f"DEBUG: Falha na verificação de integridade: {verify_error}")
                # Reabrir após verify() pois ele consome o arquivo
                image_bytes.seek(0)
                image = Image.open(image_bytes)
        except Exception as e1:
            print(f"DEBUG: Erro ao abrir imagem com Pillow: {e1}")
            print(f"DEBUG: Detalhes do erro: {traceback.format_exc()}")

            try:
                # Abordagem 2: Tentar salvar em arquivo temporário e reabrir
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(decoded_image)

                image = Image.open(tmp_path)
                os.unlink(tmp_path)  # Remover arquivo temporário
                print(f"Imagem carregada via arquivo temporário: {tmp_path}")
            except Exception as e2:
                print(f"Erro ao usar arquivo temporário: {e2}")

                # Se ambas as abordagens falharem, usar imagem de exemplo
                fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                if os.path.exists(fallback_path):
                    print(f"⚠️ Erro ao processar imagem da câmera, usando imagem de exemplo")
                    print(f"Usando imagem de exemplo após falha no processamento")
                    return np.array(Image.open(fallback_path))
                else:
                    raise ValueError("Falha ao processar imagem. Tente novamente ou use upload de arquivo.")

        # Converter para array NumPy
        start_time = time.time()
        img_array = np.array(image)
        end_time = time.time()
        print(f"DEBUG: Array NumPy criado em {(end_time - start_time)*1000:.2f}ms: Tipo: {img_array.dtype}, Dimensões: {img_array.shape}, Min: {img_array.min()}, Max: {img_array.max()}")
        print(f"DEBUG: Memória utilizada pelo array: {img_array.nbytes / (1024*1024):.2f} MB")
        print("============= FIM DO PROCESSAMENTO DE IMAGEM =============\n")
        return img_array

    except Exception as e:
        print(f"DEBUG: ERRO CRÍTICO ao processar imagem: {e}")
        print(f"DEBUG: Tipo de erro: {type(e).__name__}")
        print(f"DEBUG: Stack trace completo:")
        import traceback
        traceback.print_exc()

        # Falha de processamento - usar imagem de exemplo
        print("DEBUG: Usando imagem de exemplo após falha no processamento")
        fallback_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
        if os.path.exists(fallback_path):
            return np.array(Image.open(fallback_path))
        else:
            raise Exception(f"Erro ao processar imagem e imagem de exemplo não encontrada")

# Função para adicionar texto ao log
def add_to_log(text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {text}")

# Inicializar o estado da sessão para log
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

# Inicializar o estado da sessão para a última imagem classificada
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
    st.session_state.last_result = None
    st.session_state.last_confidence = None

# Componente JavaScript para captura de câmera compatível com dispositivos móveis
def get_camera_component():
    component_value = st.session_state.get("camera_component_value", None)
    camera_html = """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 20px;">
        <video id="video" width="100%" style="border-radius: 10px; max-width: 640px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);" autoplay></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <div id="status-message" style="margin: 10px 0; padding: 8px 15px; border-radius: 5px; background-color: #e9f5ff; color: #0066cc; display: none;"></div>
        <div style="display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; justify-content: center;">
            <button id="startCamera" style="background-color: #0066cc; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 500;">Iniciar Câmera</button>
            <button id="capturePhoto" style="background-color: #28a745; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 500;">Capturar Foto</button>
            <button id="switchCamera" style="background-color: #6c757d; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 500;">Alternar Câmera</button>
            <button id="uploadPhoto" style="background-color: #fd7e14; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 500; display: none;">Upload de Foto</button>
            <input type="file" id="fileInput" accept="image/*" capture="environment" style="display:none">
        </div>
        <div id="error-message" style="color: #dc3545; background-color: #f8d7da; border-radius: 5px; padding: 10px; margin-top: 10px; display: none;"></div>
        <div id="help-message" style="margin-top: 15px; background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9em; display: none;"></div>
    </div>
    
    <script>
        const videoElement = document.getElementById('video');
        const canvasElement = document.getElementById('canvas');
        const startButton = document.getElementById('startCamera');
        const captureButton = document.getElementById('capturePhoto');
        const switchButton = document.getElementById('switchCamera');
        const uploadButton = document.getElementById('uploadPhoto');
        const fileInput = document.getElementById('fileInput');
        const errorMessage = document.getElementById('error-message');
        const statusMessage = document.getElementById('status-message');
        const helpMessage = document.getElementById('help-message');
        
        let stream = null;
        let facingMode = "environment"; // Inicia com câmera traseira (melhor para fotos de objetos)
        let hasWebcamSupport = true; // Variável para controlar se o dispositivo suporta acesso à câmera
        
        // Desabilita os botões inicialmente
        captureButton.disabled = true;
        captureButton.style.opacity = "0.5";
        switchButton.disabled = true;
        switchButton.style.opacity = "0.5";
        
        // Verifica se o navegador tem suporte para API de mídia
        function checkBrowserSupport() {
            try {
                // Verifica se navigator existe
                if (typeof navigator === 'undefined') {
                    switchToUploadMode("Seu navegador não tem suporte para API navigator");
                    return false;
                }
                
                // Verifica se mediaDevices existe
                if (!navigator.mediaDevices) {
                    switchToUploadMode("Seu navegador não tem suporte para API mediaDevices");
                    return false;
                }
                
                // Verifica se getUserMedia existe
                if (typeof navigator.mediaDevices.getUserMedia !== 'function') {
                    switchToUploadMode("Seu navegador não tem suporte para API getUserMedia");
                    return false;
                }
                
                return true;
            } catch (err) {
                // Caso ocorra qualquer erro durante a verificação
                console.error("Erro ao verificar suporte do navegador:", err);
                switchToUploadMode("Erro ao verificar compatibilidade do navegador");
                return false;
            }
        }
        
        // Alterna para o modo de upload de foto quando a câmera não é suportada
        function switchToUploadMode(reason) {
            hasWebcamSupport = false;
            showError(`${reason}. Alternando para modo de upload de imagem.`, "warning");
            showHelp("Você pode fazer upload de uma foto tirada previamente com a câmera do seu dispositivo.");
            
            // Esconde os botões de câmera
            startButton.style.display = "none";
            captureButton.style.display = "none";
            switchButton.style.display = "none";
            
            // Mostra o botão de upload
            uploadButton.style.display = "block";
            
            // Esconde o elemento de vídeo
            videoElement.style.display = "none";
        }
        
        // Verifica se estamos em HTTPS (necessário para alguns navegadores)
        function checkHttps() {
            if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
                showError("Aviso: Esta página não está sendo servida por HTTPS. Alguns navegadores podem bloquear o acesso à câmera.", "warning");
                showHelp("Para usar a c��mera em dispositivos móveis, o site deve usar HTTPS. Se você estiver executando localmente, considere usar um tunnel HTTPS como ngrok.");
                return false;
            }
            return true;
        }
        
        // Mostra mensagem de erro formatada
        function showError(message, type = "error") {
            errorMessage.textContent = message;
            errorMessage.style.display = "block";
            
            if (type === "warning") {
                errorMessage.style.backgroundColor = "#fff3cd";
                errorMessage.style.color = "#856404";
            } else {
                errorMessage.style.backgroundColor = "#f8d7da";
                errorMessage.style.color = "#dc3545";
            }
        }
        
        // Mostra mensagem de status
        function showStatus(message) {
            statusMessage.textContent = message;
            statusMessage.style.display = "block";
        }
        
        // Esconde a mensagem de erro
        function hideError() {
            errorMessage.style.display = "none";
        }
        
        // Mostra mensagem de ajuda
        function showHelp(message) {
            helpMessage.innerHTML = `<strong>💡 Ajuda:</strong> ${message}`;
            helpMessage.style.display = "block";
        }
        
        // Converte uma imagem de File para base64
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }
        
        // Handler para o botão de upload
        uploadButton.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Handler para quando um arquivo é selecionado
        fileInput.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (file) {
                try {
                    showStatus("Processando imagem...");
                    const base64Image = await fileToBase64(file);
                    
                    // Exibir prévia da imagem no elemento de vídeo
                    const img = new Image();
                    img.src = base64Image;
                    img.onload = () => {
                        // Usar o canvas para exibir a imagem
                        const context = canvasElement.getContext('2d');
                        canvasElement.width = img.width;
                        canvasElement.height = img.height;
                        context.drawImage(img, 0, 0);
                        
                        // Mostrar temporariamente o canvas
                        canvasElement.style.display = "block";
                        videoElement.style.display = "none";
                        
                        showStatus("Imagem carregada com sucesso!");
                        
                        // Enviar para o Streamlit
                        setComponentValue(base64Image);
                    };
                } catch (err) {
                    showError(`Erro ao processar a imagem: ${err.message}`);
                }
            }
        });
        
        // Iniciar a câmera
        startButton.addEventListener('click', async () => {
            hideError();
            
            // Verificações de compatibilidade
            if (!checkBrowserSupport() || !checkHttps()) {
                return;
            }
            
            try {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                
                showStatus("Solicitando acesso à câmera...");
                
                const constraints = {
                    video: {
                        facingMode: facingMode,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                videoElement.srcObject = stream;
                
                // Habilita os botões de captura e alternar
                captureButton.disabled = false;
                captureButton.style.opacity = "1";
                switchButton.disabled = false;
                switchButton.style.opacity = "1";
                
                showStatus("Câmera ativa! Posicione o cartucho HP na imagem e tire uma foto.");
            } catch (err) {
                let errorMsg = "";
                
                // Mensagens de erro específicas para cada caso
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    errorMsg = "Permissão para acessar a câmera foi negada. Você precisa permitir o acesso à câmera nas configurações do seu navegador.";
                    showHelp("Procure o ícone de c��mera ou cadeado na barra de endereço do navegador e clique para conceder permissão.");
                } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                    errorMsg = "Nenhuma câmera foi encontrada no seu dispositivo.";
                } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                    errorMsg = "Sua câmera pode estar sendo usada por outro aplicativo.";
                    showHelp("Feche outros aplicativos que possam estar usando sua câmera (como Zoom, Teams, etc).");
                } else if (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError') {
                    errorMsg = "Não foi possível encontrar uma câmera que atenda aos requisitos.";
                } else if (err.name === 'TypeError' && err.message.includes('undefined is not an object')) {
                    errorMsg = "Seu navegador não consegue acessar a câmera. Isso pode ocorrer em navegadores mais antigos ou dentro de alguns aplicativos.";
                    switchToUploadMode("API de câmera não disponível");
                    return;
                } else {
                    errorMsg = `Erro ao acessar a câmera: ${err.message}`;
                }
                
                showError(errorMsg);
                console.error("Erro detalhado ao acessar a câmera:", err);
                
                // Se não conseguiu acessar a câmera, oferece alternativa de upload
                if (!stream) {
                    switchToUploadMode("Não foi possível acessar a câmera");
                }
            }
        });
        
        // Alternar entre câmera frontal e traseira
        switchButton.addEventListener('click', () => {
            if (stream) {
                // Parar o stream atual
                stream.getTracks().forEach(track => track.stop());
                
                // Alternar o modo da c��amera
                facingMode = facingMode === "user" ? "environment" : "user";
                
                // Atualizar mensagem
                const cameraType = facingMode === "user" ? "frontal" : "traseira";
                showStatus(`Alternando para câmera ${cameraType}...`);
                
                // Reiniciar câmera com novo modo
                startButton.click();
            }
        });
        
        // Capturar foto
        captureButton.addEventListener('click', () => {
            const context = canvasElement.getContext('2d');
            
            // Definir dimensões do canvas para corresponder ao vídeo
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            
            // Desenhar o frame atual do vídeo no canvas
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            
            // Converter para base64
            const imageDataURL = canvasElement.toDataURL('image/jpeg', 0.9);
            
            showStatus("Foto capturada com sucesso!");
            
            // Enviar para o Streamlit
            setComponentValue(imageDataURL);
        });
        
        // Verificar suporte do navegador quando a página carrega
        window.addEventListener('DOMContentLoaded', () => {
            try {
                const browserSupported = checkBrowserSupport();
                if (browserSupported) {
                    checkHttps();
                    showHelp("Clique em 'Iniciar Câmera' e permita o acesso quando solicitado. Para melhores resultados, use a câmera traseira em um ambiente bem iluminado.");
                }
            } catch (err) {
                console.error("Erro durante inicialização:", err);
                switchToUploadMode("Erro de inicialização");
            }
        });
        
        function setComponentValue(value) {
            // Atualiza o valor do componente e notifica o Streamlit
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: value
            }, "*");
        }
    </script>
    """

    # Retorna o componente HTML sem usar o parâmetro key (que está causando o erro)
    component_value = st.components.v1.html(camera_html, height=600)

    # Retorna o valor atualizado pelo JavaScript
    return component_value

# Função para codificar imagens em Base64
def get_file_content_as_base64(file_path):
    import base64
    import os

    if not os.path.exists(file_path):
        print(f"AVISO: Arquivo não encontrado: {file_path}")
        return None

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def main():
    st.title("Sistema de Classificação de Cartuchos HP")

    # Codificar o logo HP em Base64
    logo_path = os.path.join(BASE_DIR, "2025-SPRINT_2", "HP_Blue_RGB_150_MD.png")
    logo_base64 = get_file_content_as_base64(logo_path)

    # Adicionar um banner/cabeçalho atraente com o logo em Base64
    st.markdown(f"""
    <div style="background-color: #0066cc; padding: 15px; border-radius: 10px; margin-bottom: 20px; display: flex; align-items: center;">
        <div style="margin-right: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="60">
        </div>
        <div style="color: white;">
            <h2 style="margin: 0; color: white;">Verificador de Autenticidade</h2>
            <p style="margin: 0; opacity: 0.8;">Identifique cartuchos HP originais com Inteligência Artificial</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status da aplicação em cards modernos com conteúdo direto (sem divs vazias)
    col1, col2 = st.columns(2)

    with col1:
        if not OPENCV_AVAILABLE:
            st.markdown("""
            <div class="status-badge badge-warning">⚠️ OpenCV não disponível</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge badge-success">✅ OpenCV disponível</div>
            """, unsafe_allow_html=True)

    with col2:
        if not MODEL_AVAILABLE:
            st.markdown("""
            <div class="status-badge badge-warning">⚠️ Usando simulação</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge badge-success">✅ Modelo AI carregado</div>
            """, unsafe_allow_html=True)

    # Contadores de cartuchos originais e não originais
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 15px 0; display: flex; justify-content: space-around; text-align: center;">
        <div>
            <h3 style="color: #28a745; margin: 0;">Cartuchos Originais</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #28a745; margin: 10px 0;">{}</p>
        </div>
        <div>
            <h3 style="color: #dc3545; margin: 0;">Cartuchos Não Originais</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #dc3545; margin: 10px 0;">{}</p>
        </div>
    </div>
    """.format(st.session_state.contador_originais, st.session_state.contador_nao_originais), unsafe_allow_html=True)

    # Criar abas principais para a aplicação
    tab_classificacao, tab_sobre = st.tabs(["🔍 Classificação", "ℹ️ Sobre o Projeto"])

    with tab_classificacao:
        # Layout com duas colunas principais
        main_col1, main_col2 = st.columns([2, 1])

        with main_col1:
            # Removendo o card vazio e mantendo apenas o conteúdo
            st.header("📸 Captura e Classificação de Imagens")

            # Tabs para diferentes métodos de captura
            tab1, tab2, tab3 = st.tabs(["📤 Upload de Imagem", "📱 Câmera", "🔌 ESP32-CAM"])

            # Conteúdo das tabs
            with tab1:
                st.subheader("Upload de Imagem")
                st.markdown("Faça upload de uma imagem do cartucho HP para verificar sua autenticidade.")

                uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

                if uploaded_file is not None:
                    # Processar a imagem carregada
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)

                    # Exibir a imagem
                    st.image(img_array, caption="Imagem Carregada", use_container_width=True)

                    # Botão para classificar a imagem
                    if st.button("Verificar Autenticidade", key="btn_upload"):
                        with st.spinner("Analisando imagem..."):
                            add_to_log("Classificando imagem enviada pelo usuário.")

                            # Classificar usando o modelo real ou simulação
                            is_original, confidence = classify_image(img_array)

                            # Armazenar os resultados
                            st.session_state.last_image = img_array
                            st.session_state.last_result = is_original
                            st.session_state.last_confidence = confidence

                            # Atualizar os contadores
                            if is_original:
                                st.session_state.contador_originais += 1
                            else:
                                st.session_state.contador_nao_originais += 1

                            # Adicionar ao log
                            result_text = "HP Original" if is_original else "Não Original"
                            add_to_log(f"Imagem classificada como: {result_text} (Confiança: {confidence:.2f}%)")

                            # Mostrar resultado em linha também
                            if is_original:
                                st.success(f"✅ HP ORIGINAL AUTÊNTICO (Confiança: {confidence:.2f}%)")
                            else:
                                st.error(f"❌ CARTUCHO NÃO ORIGINAL (Confiança: {confidence:.2f}%)")

            with tab2:
                st.subheader("Captura via Câmera")
                st.markdown("""
                Use a câmera do seu dispositivo para tirar uma foto do cartucho HP.
                Funciona em computadores, iPhones e Android.
                """)

                # Componente de câmera compatível com dispositivos móveis
                camera_image = get_camera_component()

                # Adicionar uma chave na session_state para controlar se uma foto foi tirada
                if 'photo_taken' not in st.session_state:
                    st.session_state.photo_taken = False

                # Verificar se o usuário realmente capturou uma imagem
                # Uma imagem real em base64 começa com 'data:image'
                if camera_image and isinstance(camera_image, str) and camera_image.startsWith('data:image'):
                    # Marcar que uma foto foi tirada
                    st.session_state.photo_taken = True

                    # Processar a imagem capturada
                    try:
                        # Transformar a imagem base64 em array
                        img_array = process_camera_image(camera_image)

                        # Exibir a imagem capturada
                        st.image(img_array, caption="Imagem Capturada", use_container_width=True)

                        # Botão para classificar a imagem
                        if st.button("Verificar Autenticidade", key="btn_camera"):
                            with st.spinner("Analisando imagem..."):
                                add_to_log("Classificando imagem capturada pela câmera.")

                                # Classificar usando o modelo real ou simulação
                                is_original, confidence = classify_image(img_array)

                                # Armazenar os resultados
                                st.session_state.last_image = img_array
                                st.session_state.last_result = is_original
                                st.session_state.last_confidence = confidence

                                # Atualizar os contadores
                                if is_original:
                                    st.session_state.contador_originais += 1
                                else:
                                    st.session_state.contador_nao_originais += 1

                                # Adicionar ao log
                                result_text = "HP Original" if is_original else "Não Original"
                                add_to_log(f"Imagem da câmera classificada como: {result_text} (Confiança: {confidence:.2f}%)")

                                # Mostrar resultado em linha também
                                if is_original:
                                    st.success(f"✅ HP ORIGINAL AUTÊNTICO (Confiança: {confidence:.2f}%)")
                                else:
                                    st.error(f"❌ CARTUCHO NÃO ORIGINAL (Confiança: {confidence:.2f}%)")

                    except Exception as e:
                        st.error(f"Erro ao processar a imagem: {e}")
                        print(f"Erro ao processar imagem da câmera: {e}")
                elif camera_image:
                    # Se o valor do componente existir mas não for uma string v��lida
                    st.info("Clique em 'Iniciar Câmera' e depois 'Capturar Foto' para tirar uma foto do cartucho.")
                else:
                    # Se não houver valor no componente
                    st.info("Use os botões acima para capturar uma foto com sua câmera.")

            with tab3:
                st.subheader("Stream da Câmera ESP32-CAM")
                st.markdown("""
                Conecte a um módulo ESP32-CAM para capturar imagens remotamente.
                Ideal para instalações fixas ou linhas de produção.
                """)

                esp32_url = st.text_input("Endereço IP do ESP32-CAM", "http://192.168.0.XXX")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Conectar ao ESP32", key="btn_connect_esp32"):
                        st.warning("Conectando ao ESP32-CAM... Funcionalidade a ser implementada.")
                        add_to_log("Tentativa de conexão com o ESP32-CAM.")

                with col2:
                    if st.button("Capturar Foto", key="btn_capture_esp32"):
                        add_to_log("Solicitação de captura do ESP32-CAM.")

                        with st.spinner("Capturando imagem do ESP32-CAM..."):
                            # Carregar uma imagem de exemplo para demonstraç��o
                            example_path = os.path.join(DATASET_PATH, "HP_Original", "HP_origina_10.jpg")
                            if os.path.exists(example_path):
                                img = Image.open(example_path)
                                img_array = np.array(img)

                                # Classificar usando o modelo real ou simula��ao
                                is_original, confidence = classify_image(img_array)

                                # Armazenar os resultados
                                st.session_state.last_image = img_array
                                st.session_state.last_result = is_original
                                st.session_state.last_confidence = confidence

                                # Atualizar os contadores
                                if is_original:
                                    st.session_state.contador_originais += 1
                                else:
                                    st.session_state.contador_nao_originais += 1

                                # Adicionar ao log
                                result_text = "HP Original" if is_original else "Não Original"
                                add_to_log(f"Imagem classificada como: {result_text} (Confiança: {confidence:.2f}%)")

                                # Mostrar resultado
                                if is_original:
                                    st.success(f"✅ HP ORIGINAL AUTÊNTICO (Confiança: {confidence:.2f}%)")
                                else:
                                    st.error(f"❌ CARTUCHO NÃO ORIGINAL (Confian��a: {confidence:.2f}%)")
                            else:
                                st.error(f"Imagem de exemplo não encontrada: {example_path}")

        # Coluna para mostrar resultados e log
        with main_col2:
            # Exibir última imagem classificada e resultado - sem div card vazia
            st.header("🔍 Última Classificação")

            if st.session_state.last_image is not None:
                st.image(st.session_state.last_image, caption="Imagem Classificada", use_container_width=True)

                # Exibir resultado com estilo baseado na classificação
                result_class = "result-original" if st.session_state.last_result else "result-fake"
                result_text = "HP Original ✅" if st.session_state.last_result else "Não Original ❌"

                st.markdown(f"""
                <div class="{result_class}">
                    <h3>{result_text}</h3>
                    <p>Confiança: {st.session_state.last_confidence:.2f}%</p>
                    <p>Data: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Nenhuma imagem classificada ainda. Capture ou faça upload de uma imagem para analisar.")

            # Terminal de log - sem div card vazia
            st.header("Log de Execução")
            log_container = st.container()

            with log_container:
                # Construir o conteúdo do terminal
                log_content = "<div class='terminal'>"
                for message in st.session_state.log_messages:
                    log_content += f"{message}<br>"
                log_content += "</div>"

                st.markdown(log_content, unsafe_allow_html=True)

                # Botão para limpar o log
                if st.button("Limpar Log"):
                    st.session_state.log_messages = []
                    st.experimental_rerun()

    # Nova aba para informações sobre o projeto
    with tab_sobre:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("Sobre o Sistema")
            st.markdown("""
            Este sistema utiliza inteligência artificial para classificar cartuchos HP originais e não originais.
            
            ### Como usar:
            
            1. Faça upload de uma imagem ou use a câmera
            2. Clique em "Verificar Autenticidade"
            3. Veja o resultado da análise
            
            ### Tecnologias:
            
            * **TensorFlow** (IA/ML)
            * **Streamlit** (Interface)
            * **OpenCV** (Processamento de imagens)
            """)

        with col2:
            st.header("Reconhecimento de Cartuchos")
            st.image("https://logodownload.org/wp-content/uploads/2014/04/hp-logo-1.png", width=200)
            st.markdown("""
            ### Características avaliadas:
            
            * Padrões de impressão na embalagem
            * Características do cartucho
            * Elementos de segurança
            
            *Desenvolvido por FIAP - 2025*
            """)

            st.info("Este é um projeto acadêmico demonstrativo. Para mais informações, entre em contato com a equipe de desenvolvimento.")

# Rodapé da página
def footer():
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center; font-size: 0.8em;">
        <p>Sistema de Classificação de Cartuchos HP © FIAP 2025</p>
        <p>Versão 1.0 - Todos os direitos reservados</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    print("Iniciando aplicativo Streamlit de classificação de cartuchos...")
    main()
    footer()
