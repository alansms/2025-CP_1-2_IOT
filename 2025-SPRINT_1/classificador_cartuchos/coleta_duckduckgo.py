# Script: coleta_duckduckgo.py (compatível com versões recentes do duckduckgo-search)
# Requisitos: pip install duckduckgo-search requests Pillow

import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

def baixar_imagens(termo_busca, pasta_destino, limite=30):
    ddgs = DDGS()
    resultados = list(ddgs.images(termo_busca, max_results=limite))
    if not resultados:
        print(f"Nenhum resultado encontrado para: {termo_busca}")
        return

    os.makedirs(pasta_destino, exist_ok=True)
    for i, img in enumerate(resultados):
        url = img['image']
        try:
            resposta = requests.get(url, timeout=10)
            imagem = Image.open(BytesIO(resposta.content)).convert('RGB')
            caminho_arquivo = os.path.join(pasta_destino, f"{termo_busca[:10].replace(' ', '_')}_{i+1}.jpg")
            imagem.save(caminho_arquivo)
            print(f"✅ Imagem salva: {caminho_arquivo}")
        except Exception as e:
            print(f"⚠️ Erro ao baixar imagem {i+1}: {e}")

# Baixar imagens de cartuchos HP originais isolados
baixar_imagens("HP original ink cartridge box isolated", "dataset/HP_Original", limite=30)

# Baixar imagens de cartuchos de outras marcas (não HP)
baixar_imagens("Canon ink cartridge product", "dataset/Outros", limite=15)
baixar_imagens("Epson ink cartridge product", "dataset/Outros", limite=15)

print("✅ Coleta finalizada!")