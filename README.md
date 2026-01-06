# 🎙️ Corvo - AI Voice Assistant & Computer Vision

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)
![Groq](https://img.shields.io/badge/LLM-Groq_API-purple)

## 📄 Descrição
O **Corvo** é um assistente virtual inteligente desenvolvido em Python que vai além dos comandos de voz tradicionais. Ele integra **Inteligência Artificial Generativa (LLM)** para conversação natural e módulos de **Visão Computacional** para interagir com o mundo real.

O sistema é capaz de reconhecer rostos, classificar imagens, ler textos (OCR) e gerenciar tarefas do dia a dia, tudo controlado por comandos de voz em português.

## 🚀 Funcionalidades Principais

### 🧠 Inteligência Artificial (LLM)
* **Integração com Groq API:** Utiliza o modelo `openai/gpt-oss-120b` para responder perguntas complexas e manter conversas naturais.
* **Contexto de Sistema:** Personalidade configurada ("Você é o Corvo...") para respostas objetivas.

### 👁️ Visão Computacional
* **Reconhecimento Facial:** Sistema treinado com algoritmo LBPH (OpenCV) para identificar usuários cadastrados (Ex: Caique, Allex, Diego).
* **Classificação de Imagens:** Modelo de Deep Learning (TensorFlow/Keras) para identificar animais (Bird, Cat, Dog) em tempo real via webcam ou arquivo.
* **OCR (Leitura de Texto):** Utiliza `easyocr` para ler e verbalizar textos apontados para a câmera.

### 🛠️ Ferramentas & Automação
* **Agenda Inteligente:** Adiciona, lista e remove compromissos por voz (salvos em JSON).
* **Automação de Desktop:** Tira screenshots da tela por comando de voz.
* **Informações em Tempo Real:**
    * Previsão do Tempo (via wttr.in).
    * Cotação de Moedas (Dólar, Euro, Bitcoin via AwesomeAPI).
    * Data e Hora atualizadas.
* **Entretenimento:** Toca músicas buscando diretamente no YouTube.

## 💻 Tecnologias Utilizadas
* **Linguagem:** Python
* **Voz:** `speech_recognition` (STT), `gTTS` e `pygame` (TTS).
* **Visão:** `opencv-python`, `tensorflow`, `easyocr`.
* **Conectividade:** `requests` (APIs REST).

## ⚙️ Configuração e Instalação

### Pré-requisitos
1. Python instalado.
2. Uma chave de API da **Groq Cloud**.

### Instalação
```bash
# Clone o repositório

# Instale as dependências
pip install -r requirements.txt
