# Classificador de Cartuchos HP - Captura Contínua com ESP32-CAM

![Logo HP](2025-SPRINT_2/HP_Blue_RGB_150_MD.png)

Este repositório contém o sistema de classificação de cartuchos HP, desenvolvido para a SPRINT 2 do projeto. O sistema utiliza o ESP32-CAM para captura contínua de imagens e um servidor Python para processamento e classificação em tempo real, identificando cartuchos originais HP de outras marcas.

## 🔗 Demonstração Online

Acesse a aplicação em funcionamento através do link:
**[https://2025-cp1-2iot-6fcbbvcwsskqmjpacwxwpt.streamlit.app](https://2025-cp1-2iot-6fcbbvcwsskqmjpacwxwpt.streamlit.app)**

## 📋 Descrição do Projeto

Este sistema realiza a captura automática de imagens de cartuchos de impressora via ESP32-CAM e as transmite para um computador via rede local, onde são processadas e classificadas utilizando um modelo de Deep Learning treinado na Sprint 1. Os resultados da classificação são exibidos em uma interface web Streamlit, incluindo estatísticas de detecção e registros de processamento.

## 🚀 Recursos Implementados

### 🔌 Captura Contínua com ESP32-CAM
- **Captura Periódica**: O ESP32-CAM captura imagens automaticamente a cada 3 segundos
- **Transmissão via Rede Local**: Envio das imagens por HTTP POST para o servidor Python
- **Formato Otimizado**: Imagens enviadas em formato JPEG ou codificadas em base64
- **Operação Autônoma**: Sistema funciona em rede WiFi local sem necessidade de internet

### 🖥️ Servidor de Processamento
- **Recepção de Imagens**: Servidor Flask para receber as imagens do ESP32-CAM
- **Processamento em Tempo Real**: Pré-processamento e inferência imediatos após recebimento
- **Registro de Atividades**: Log detalhado com horário de captura e resultados

### 🔍 Sistema de Classificação de Imagens
- **Modelo CNN Pré-treinado**: Utiliza o modelo treinado na Sprint 1 para classificação
- **Processamento de Imagem**: Algoritmos que preservam a proporção original das imagens
- **Alta Performance**: Otimizado para classificação rápida em tempo real

### 📱 Interface de Usuário
- **Dashboard Streamlit**: Interface web para visualização dos resultados em tempo real
- **Visualização de Resultados**: Exibição clara dos resultados com níveis de confiança
- **Contadores Estatísticos**: Acompanhamento do número de cartuchos originais e não originais
- **Terminal de Log**: Registro detalhado das operações para rastreamento e depuração

## 🛠️ Recursos Técnicos

### 📷 ESP32-CAM
- **Captura Automática**: Programado para capturar imagens em intervalos regulares
- **Envio HTTP**: Implementação otimizada para envio confiável das imagens
- **Reconexão Automática**: Recuperação de conexão em caso de falhas na rede
- **Configuração Flexível**: Parâmetros ajustáveis como intervalo, resolução e qualidade

### 💻 Processamento de Imagem
- **Preservação de Proporção**: Algoritmo que mantém a proporção original das imagens
- **Detecção Automática de Formato**: Identificação do formato da imagem pelos bytes iniciais
- **Tratamento de Erros Robusto**: Sistema de fallback para garantir operação contínua
- **Verificação de Integridade**: Validação da integridade das imagens recebidas

### 🧠 Modelo de Classificação
- **Sistema de Zonas de Confiança**: Classificação baseada em zonas de confiança para maior precisão
- **Penalidade Adaptativa**: Ajustes de confiança para reduzir falsos positivos
- **Análise de Performance**: Métricas de tempo de inferência e confiabilidade do sistema

## 📋 Tecnologias Utilizadas

- **Hardware**: ESP32-CAM
- **Firmware**: Arduino IDE
- **Backend**: Python (Flask)
- **Frontend**: Streamlit
- **Processamento**: OpenCV, NumPy
- **Modelo de IA**: TensorFlow/Keras
- **Comunicação**: HTTP/RESTful

## 🚀 Como Executar

### Requisitos
- ESP32-CAM
- Python 3.8+
- Rede WiFi local
- Bibliotecas listadas em `requirements.txt`

### Preparação do ESP32-CAM
1. Abra o arquivo `esp32_code/esp32_camera.ino` no Arduino IDE
2. Configure as credenciais WiFi e o endereço IP do servidor
3. Faça o upload do firmware para o ESP32-CAM

### Configuração do Servidor
```bash
# Clonar o repositório
git clone https://github.com/alansms/2025-CP_1-2_IOT.git
cd 2025-CP_1-2_IOT/2025-SPRINT_2/streamlit_app

# Instalar dependências
pip install -r requirements.txt

# Iniciar o servidor de recepção e a interface
streamlit run app.py
```

### Uso
1. Inicie o ESP32-CAM conectado à mesma rede WiFi do computador
2. Abra a interface Streamlit no navegador (geralmente em http://localhost:8501)
3. Posicione o ESP32-CAM para visualizar os cartuchos de impressora
4. A classificação acontecerá automaticamente a cada nova imagem recebida
5. Visualize os resultados e estatísticas na interface

## 📊 Estrutura do Projeto

```
2025-CP_1-2_IOT/
├── README.md                         # Documentação principal
├── 2025-SPRINT_1/                    # Código da Sprint 1 (Treinamento do modelo)
└── 2025-SPRINT_2/                    # Código da Sprint 2 (Sistema ESP32 + Interface)
    ├── HP_Blue_RGB_150_MD.png        # Logo HP
    ├── esp32_code/                   # Código para ESP32-CAM
    │   └── esp32_camera.ino          # Firmware com captura periódica e envio HTTP
    └── streamlit_app/                # Aplicação principal
        ├── app.py                    # Servidor Flask + Interface Streamlit
        └── requirements.txt          # Dependências do projeto
```

## 📈 Métricas e Desempenho

- **Tempo Médio de Inferência**: ~200ms por imagem (depende do hardware)
- **Confiabilidade da Transmissão**: >95% em redes locais estáveis
- **Acurácia do Modelo**: ~90% na identificação de cartuchos originais HP
- **Intervalo de Captura**: Configurável, padrão de 3 segundos
- **Consumo de Recursos**: ~120MB de RAM durante operação

## 👥 Participantes do Projeto

- **André Rovai Andrade Xavier Junior** - RM555848@fiap.com.br
- **Antonio Vinicius Vicentini Liberato** - RM558014@fiap.com.br
- **Alan de Souza Maximiano da Silva** - RM557088@fiap.com.br
- **Leonardo Zago Garcia Ferreira** - RM558691@fiap.com.br
- **Renan de França Gonçalves** - RM558413@fiap.com.br
- **Thiago Almança da Silva** - RM558108@fiap.com.br

## 📌 Próximos Passos

- Implementação de banco de dados para armazenamento histórico de classificações
- Sistema de alertas por e-mail ou notificações push
- Interface de administração para configuração remota do ESP32-CAM
- Suporte para múltiplos dispositivos ESP32-CAM
- Expansão do modelo para classificação de outros tipos de cartuchos

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais e de demonstração.

---

Desenvolvido por FIAP © 2025
