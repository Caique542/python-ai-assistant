# -*- coding: utf-8 -*-

# --- Bibliotecas do Assistente Master e Agenda ---
import re
import math
# from zoneinfo import ZoneInfo  <-- Mantido comentado para evitar erros de fuso horário no Windows
import speech_recognition as sr
import requests
import datetime
import os
import numpy as np
import time
import easyocr
import pyautogui
import webbrowser
from urllib.parse import quote
import string
import random
import json

# --- Bibliotecas de Classificação Keras ---
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError

# --- Bibliotecas da API Externa (gTTS e Pygame para áudio) ---
from gtts import gTTS
import pygame  # Usado para reprodução de áudio estável

# --- CONFIGURAÇÕES GLOBAIS DE RECONHECIMENTO FACIAL ---
RECONHECEDOR_A_SER_USADO = "LBPH"
NOME_ARQUIVO_MODELO_FACIAL = f"classificador{RECONHECEDOR_A_SER_USADO}.yml"
detector_facial = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- CONFIGURAÇÕES DE CLASSIFICAÇÃO KERAS ---
CAMINHO_MODELO_KERAS = 'meu_modelo_4_classes_balanceado_grayscale.keras'
CLASS_NAMES = ['bird', 'cat', 'desconhecido', 'dog']
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
MIN_CONTOUR_AREA = 5000

# --- CONFIGURAÇÕES DA API GROQ ---
# [SEGURANÇA] Chave mascarada para upload no GitHub.
# O usuário deve inserir sua própria chave aqui ou usar variáveis de ambiente.
GROQ_API_KEY = "INSIRA_SUA_CHAVE_GROQ_AQUI" 
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"  # Modelo correto e ativo
GROQ_SYSTEM_PROMPT = "Você é o Corvo, um assistente virtual inteligente e prestativo. Responda sempre de forma clara e objetiva, em português."

# --- CONFIGURAÇÕES DA AGENDA ---
NOME_ARQUIVO_AGENDA = 'agenda.json'

# Lista de comandos para impressão no console
COMANDOS_LISTA = """
==============================
      COMANDOS DISPONÍVEIS
==============================
* AGENDA:        Diga 'abrir agenda' ou 'agenda'.
* CÁLCULO:       Diga 'calcule 5 vezes 8', '4 ao quadrado', etc.
* DATA/HORA:     Diga 'que horas são', 'qual é a data'.
* RECON. FACIAL: Diga 'iniciar reconhecimento facial'.
* CLASSIFICAR ANIMAIS: Diga 'classificar animais'
* RECON. ANIMAIS: Diga 'iniciar reconhecimento de animais'
* LEITURA OCR:   Diga 'iniciar OCR' ou 'ler texto'.
* GROQ IA:       Diga 'Ei Corvo [sua pergunta]'.
* SCREENSHOT:    Diga 'tirar screenshot [nome]'.
* TEMPO:         Diga 'previsão do tempo em [cidade]'.
* COTAÇÃO:       Diga 'cotação do euro', 'preço do bitcoin'.
* MÚSICA:        Diga 'tocar música [nome da música]'.
* SAIR:          Diga 'parar' ou 'sair'.
==============================
"""


# =========================================================================
# FUNÇÕES GLOBAIS DA AGENDA (Fora da classe)
# =========================================================================
def carregar_agenda():
    """Carrega os compromissos do arquivo JSON."""
    if not os.path.exists(NOME_ARQUIVO_AGENDA):
        return []
    try:
        with open(NOME_ARQUIVO_AGENDA, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content: return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []


def salvar_agenda(agenda):
    """Salva a lista de compromissos no arquivo JSON."""
    with open(NOME_ARQUIVO_AGENDA, 'w', encoding='utf-8') as f:
        json.dump(agenda, f, indent=4, ensure_ascii=False)


# =========================================================================
# CLASSE DE CLASSIFICAÇÃO (KERAS) - Sem alterações
# =========================================================================
class ClassificadorKeras:
    def __init__(self, tts_func):
        self.model = None
        self.falar = tts_func
        self.carregar_modelo()

    def carregar_modelo(self):
        if not os.path.exists(CAMINHO_MODELO_KERAS):
            print(f"ERRO CRÍTICO: Arquivo do modelo Keras não encontrado em {CAMINHO_MODELO_KERAS}")
            return
        try:
            self.model = tf.keras.models.load_model(CAMINHO_MODELO_KERAS, compile=False)
            print("Modelo Keras carregado com sucesso.")
        except Exception as e:
            print(f"ERRO CRÍTICO: Falha ao carregar o modelo Keras. Detalhe: {e}")
            self.model = None

    def prever_de_arquivo(self):
        if not self.model:
            self.falar("Modelo de classificação não carregado.")
            return
        caminho_imagem = input("Digite o caminho da imagem (ou arraste o arquivo para o terminal): ").strip().replace(
            "'", "").replace('"', '')
        if not os.path.exists(caminho_imagem):
            self.falar(f"Imagem não encontrada em '{os.path.basename(caminho_imagem)}'")
            return
        self.falar(f"Analisando {os.path.basename(caminho_imagem)}")
        try:
            img = image.load_img(caminho_imagem, target_size=IMG_SIZE, color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            previsao = self.model.predict(img_array, verbose=0)
            confianca = np.max(previsao)
            indice_predito = np.argmax(previsao)
            classe_predita = CLASS_NAMES[indice_predito]
            print("\n--- Resultado da Previsão ---")
            if confianca >= CONFIDENCE_THRESHOLD:
                self.falar(f"A imagem é um {classe_predita}. Confiança de {confianca * 100:.0f} por cento.")
                print(f"Confiança: {confianca * 100:.2f}%")
            else:
                self.falar(
                    f"A confiança do modelo é baixa. O modelo sugere {classe_predita}, mas com apenas {confianca * 100:.0f} por cento de confiança.")
                print(f"Confiança: {confianca * 100:.2f}% (Abaixo do limiar)")
        except UnidentifiedImageError:
            self.falar(f"Erro. O arquivo não é um formato de imagem válido.")
        except Exception as e:
            self.falar(f"Erro ao processar a imagem.")
            print(f"Detalhe do erro: {e}")

    def prever_com_webcam(self):
        if not self.model:
            self.falar("Modelo de classificação não carregado.")
            return
        cap = cv2.VideoCapture(0)
        time.sleep(1.0)
        if not cap.isOpened():
            self.falar("Não foi possível abrir a câmera para classificação.")
            return
        self.falar("Iniciando reconhecimento de animais. Pressione 'q' para sair.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                maior_contorno = max(contours, key=cv2.contourArea)
                if cv2.contourArea(maior_contorno) > MIN_CONTOUR_AREA:
                    (x, y, w, h) = cv2.boundingRect(maior_contorno)
                    roi_gray = gray[y:y + h, x:x + w]
                    resized_roi = cv2.resize(roi_gray, IMG_SIZE)
                    img_array = np.expand_dims(resized_roi, axis=-1)
                    img_array = np.expand_dims(img_array, axis=0)
                    previsao = self.model.predict(img_array, verbose=0)
                    confianca = np.max(previsao)
                    indice_predito = np.argmax(previsao)
                    classe_predita = CLASS_NAMES[indice_predito]
                    texto_resultado = f"{classe_predita} ({confianca * 100:.0f}%)"
                    cor_texto = (0, 255, 0) if confianca >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                    cor_retangulo = (0, 255, 0) if confianca >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), cor_retangulo, 2)
                    cv2.putText(frame, texto_resultado, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_texto, 2)
            cv2.imshow('Reconhecimento de Animais - Pressione "q" para sair', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        self.falar("Reconhecimento de animais finalizado.")


# =========================================================================
# CLASSE PRINCIPAL DO ASSISTENTE
# =========================================================================
class AssistenteMaster:
    def __init__(self):
        pygame.mixer.init()
        self.reconhecedor = sr.Recognizer()
        self.classificador_keras = ClassificadorKeras(self.falar)
        self.main_loop_running = True

    # =========================================================================
    # FUNÇÕES DE INTERAÇÃO POR VOZ (FALAR E OUVIR) - UNIFICADAS
    # =========================================================================
    def falar(self, texto):
        """Converte texto em áudio e o reproduz usando pygame para estabilidade."""
        try:
            print(f"Assistente: {texto}")
            tts = gTTS(text=texto, lang='pt-br')
            arquivo_temp = f"resposta_temp_{int(time.time())}_{random.randint(0, 1000)}.mp3"
            tts.save(arquivo_temp)

            pygame.mixer.music.load(arquivo_temp)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()

            if os.path.exists(arquivo_temp):
                os.remove(arquivo_temp)
        except Exception as e:
            print(f"Ocorreu um erro ao tentar falar: {e}")

    def ouvir_resposta_especifica(self, prompt_para_usuario="Ouvindo..."):
        """Captura uma resposta específica do usuário durante um diálogo."""
        with sr.Microphone() as mic:
            print(prompt_para_usuario)
            self.reconhecedor.adjust_for_ambient_noise(mic, duration=0.5)
            try:
                audio = self.reconhecedor.listen(mic, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("Não detectei nenhuma fala. Tente novamente.")
                return ""

        try:
            print("Reconhecendo...")
            texto = self.reconhecedor.recognize_google(audio, language='pt-BR').lower()
            print(f"Você disse: '{texto}'")
            return texto
        except sr.UnknownValueError:
            self.falar("Desculpe, não consegui entender o que você disse.")
            return ""
        except sr.RequestError:
            self.falar("Estou com problemas de conexão para reconhecer sua voz.")
            return ""

    # =========================================================================
    # FUNÇÃO DE COMUNICAÇÃO COM A GROQ (LLM)
    # =========================================================================
    def perguntar_groq(self, user_prompt):
        # Verifica se a chave foi configurada
        if not GROQ_API_KEY or "INSIRA_SUA_CHAVE" in GROQ_API_KEY:
            self.falar("A chave da API Groq não foi configurada. Por favor, adicione a chave no código.")
            return
        self.falar(f"Perguntando ao Corvo sobre {user_prompt[:30]}...")
        tempo_inicial = time.time()
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": GROQ_MODEL, "messages": [{"role": "system", "content": GROQ_SYSTEM_PROMPT},
                                                  {"role": "user", "content": user_prompt}], "temperature": 0.7}

        try:
            response = requests.post(GROQ_URL, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            resposta = response.json()
            if 'choices' in resposta and len(resposta['choices']) > 0:
                conteudo = resposta['choices'][0]['message']['content'].strip()
                resposta_final = conteudo.split("</think>")[-1].strip()
                print(f"\n🤖 Resposta da IA (Groq): {resposta_final}")
                self.falar(resposta_final)
            else:
                self.falar("A Groq não retornou uma resposta válida.")
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 401:
                self.falar("Ocorreu um erro de autenticação. Sua chave de API está incorreta ou inválida.")
            else:
                self.falar("Desculpe, houve um erro de comunicação com o servidor da Groq.")
        except requests.exceptions.RequestException:
            self.falar("Desculpe, houve um erro de conexão com o servidor da Groq.")
        print(f"\n⏱️ Tempo de resposta Groq: {time.time() - tempo_inicial:.2f} segundos")

    # =========================================================================
    # MÉTODOS DA AGENDA (INTEGRADOS NA CLASSE)
    # =========================================================================
    def menu_agenda(self):
        """Gerencia o menu principal da agenda por comandos de voz."""
        agenda_completa = carregar_agenda()
        self.falar("Módulo de agenda ativado.")

        while True:
            self.falar("Qual comando da agenda você deseja? Adicionar, visualizar, excluir, limpar ou voltar?")
            comando = self.ouvir_resposta_especifica("Aguardando comando da agenda...")

            if 'adicionar' in comando:
                self.adicionar_compromisso_falado(agenda_completa)
            elif 'visualizar' in comando or 'ver' in comando or 'listar' in comando:
                self.visualizar_compromissos_falado(agenda_completa)
            elif 'excluir' in comando or 'remover' in comando:
                self.excluir_compromisso_falado(agenda_completa)
            elif 'limpar' in comando:
                self.limpar_agenda_falado(agenda_completa)
            elif 'voltar' in comando or 'sair' in comando or 'encerrar' in comando:
                self.falar("Retornando ao menu principal.")
                break
            elif comando:
                self.falar("Comando da agenda não reconhecido.")

    def adicionar_compromisso_falado(self, agenda):
        """Conduz um diálogo para adicionar um novo compromisso."""
        self.falar("Ok, vamos adicionar um novo compromisso.")

        while True:
            self.falar("Qual é o nome do evento?")
            evento = self.ouvir_resposta_especifica("Diga o nome do evento...")
            if evento: break

        ano_atual = datetime.datetime.now().year
        self.falar(f"O compromisso é para este ano, {ano_atual}?")
        resposta_ano = self.ouvir_resposta_especifica("Diga 'sim' ou 'não'...")

        if 'sim' in resposta_ano:
            ano = ano_atual
            self.falar(f"Ok, agendando para {ano_atual}.")
        else:
            self.falar("Entendido.")
            while True:
                self.falar("Então, para qual ano seria o evento?")
                ano_texto = self.ouvir_resposta_especifica("Diga o número do ano...")
                try:
                    ano = int(ano_texto)
                    if ano >= ano_atual:
                        break
                    else:
                        self.falar(f"O ano não pode ser no passado. Por favor, diga um ano a partir de {ano_atual}.")
                except (ValueError, TypeError):
                    self.falar("Não entendi o número. Por favor, diga apenas o ano.")

        while True:
            self.falar("Qual é o mês?")
            mes_texto = self.ouvir_resposta_especifica("Diga o número do mês...")
            try:
                mes = int(mes_texto)
                if 1 <= mes <= 12:
                    break
                else:
                    self.falar("Mês inválido. Diga um número de 1 a 12.")
            except (ValueError, TypeError):
                self.falar("Não entendi o número.")

        while True:
            self.falar("Qual é o dia?")
            dia_texto = self.ouvir_resposta_especifica("Diga o número do dia...")
            try:
                dia = int(dia_texto)
                if 1 <= dia <= 31:
                    break
                else:
                    self.falar("Dia inválido.")
            except (ValueError, TypeError):
                self.falar("Não entendi o número.")

        while True:
            self.falar("Qual a hora do compromisso? Diga apenas a hora, de 0 a 23.")
            hora_texto = self.ouvir_resposta_especifica("Diga a hora...")
            try:
                hora = int(hora_texto)
                if 0 <= hora <= 23:
                    break
                else:
                    self.falar("Hora inválida.")
            except (ValueError, TypeError):
                self.falar("Não entendi o número.")

        while True:
            self.falar("E os minutos?")
            minuto_texto = self.ouvir_resposta_especifica("Diga os minutos...")
            try:
                minuto = int(minuto_texto)
                if 0 <= minuto <= 59:
                    break
                else:
                    self.falar("Minuto inválido.")
            except (ValueError, TypeError):
                self.falar("Não entendi o número.")

        hora_str = f"{hora:02d}:{minuto:02d}"
        novo_compromisso = {'evento': evento, 'dia': dia, 'mes': mes, 'ano': ano, 'hora': hora_str}
        agenda.append(novo_compromisso)
        salvar_agenda(agenda)
        self.falar("Perfeito! Compromisso adicionado com sucesso.")

    def visualizar_compromissos_falado(self, agenda):
        self.falar("Ok, vou listar seus compromissos.")
        if not agenda:
            self.falar("Você não possui nenhum compromisso agendado.")
            return False

        agenda_ordenada = sorted(agenda, key=lambda c: (c['ano'], c['mes'], c['dia'], c['hora']))
        for i, compromisso in enumerate(agenda_ordenada, start=1):
            info = f"Compromisso {i}: {compromisso['evento']} no dia {compromisso['dia']} do {compromisso['mes']} de {compromisso['ano']}, às {compromisso['hora']}"
            self.falar(info)
        return True

    def excluir_compromisso_falado(self, agenda):
        self.falar("Certo, vamos excluir um compromisso.")
        if not self.visualizar_compromissos_falado(agenda):
            return

        while True:
            self.falar("Qual o número do compromisso que você deseja excluir?")
            numero_texto = self.ouvir_resposta_especifica("Diga o número do compromisso...")
            try:
                agenda_ordenada = sorted(agenda, key=lambda c: (c['ano'], c['mes'], c['dia'], c['hora']))
                escolha_num = int(numero_texto)
                index_alvo = escolha_num - 1
                if 0 <= index_alvo < len(agenda_ordenada):
                    compromisso_a_excluir = agenda_ordenada[index_alvo]
                    break
                else:
                    self.falar(f"Número inválido. Por favor, diga um número entre 1 e {len(agenda_ordenada)}.")
            except (ValueError, TypeError):
                self.falar("Não entendi o número. Tente novamente.")

        self.falar(
            f"Você tem certeza que quer excluir o evento: {compromisso_a_excluir['evento']}? Responda sim ou não.")
        confirmacao = self.ouvir_resposta_especifica("Diga 'sim' ou 'não'...")

        if 'sim' in confirmacao:
            agenda.remove(compromisso_a_excluir)
            salvar_agenda(agenda)
            self.falar("Compromisso excluído.")
        else:
            self.falar("Ok, operação cancelada.")

    def limpar_agenda_falado(self, agenda):
        self.falar("Atenção! Este comando apaga todos os compromissos. Você tem certeza? Responda sim ou não.")
        confirmacao = self.ouvir_resposta_especifica("Diga 'sim' ou 'não'...")

        if 'sim' in confirmacao:
            agenda.clear()
            salvar_agenda(agenda)
            self.falar("Pronto. Todos os compromissos foram apagados.")
        else:
            self.falar("Operação cancelada.")

    # =========================================================================
    # FUNÇÕES DE CÁLCULO, DATA/HORA E OUTRAS UTILIDADES
    # =========================================================================
    def calculadora(self, user_query):
        query = user_query.lower()
        numeros_por_extenso = {'um': '1', 'uma': '1', 'dois': '2', 'três': '3', 'quatro': '4', 'cinco': '5',
                               'seis': '6', 'sete': '7', 'oito': '8', 'nove': '9', 'dez': '10'}
        for palavra, numero in numeros_por_extenso.items():
            query = query.replace(palavra, numero)
        query = query.replace("qual é a qual é a", "qual é a")
        replacements = {' ao quadrado': '**2', ' elevado a ': '**', 'raiz de': 'sqrt', ' quanto é': '', ' calcule': '',
                        ' quanto dá': '', ' mais ': '+', ' menos ': '-', ' vezes ': '*', ' x ': '*',
                        ' dividido por ': '/', ',': '.'}
        for word, symbol in replacements.items():
            query = query.replace(word, symbol)
        query = query.strip()
        query = re.sub(r'sqrt\s*(\d+\.?\d*)', r'math.sqrt(\1)', query)
        query = re.sub(r'[^\d().+*/-]', '', query)
        try:
            if not query: return "Expressão de cálculo vazia."
            if query.endswith(('+', '-', '*', '/')): return "Parece que a expressão está incompleta."
            safe_scope = {'math': math}
            result = eval(query, {"__builtins__": None}, safe_scope)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            elif isinstance(result, float):
                result = round(result, 2)
            self.falar(f"O resultado é {result}.")
        except Exception as e:
            self.falar("Desculpe, não consegui calcular a expressão.")

    def data_hora(self):
        try:
            # CORREÇÃO: Usa datetime.now() direto, sem forçar timezone que pode não estar instalada
            agora = datetime.datetime.now()
            self.falar(agora.strftime("Agora são %H horas e %M minutos."))
        except Exception as e:
            print(f"Erro ao verificar horário: {e}")
            self.falar("Desculpe, ocorreu um erro ao verificar o horário.")

    def data_completa(self):
        try:
            # CORREÇÃO: Usa datetime.now() direto
            agora_em_sp = datetime.datetime.now()
            response = agora_em_sp.strftime("Hoje é dia %d de %B de %Y.")
            months = {'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril', 'May': 'Maio',
                      'June': 'Junho', 'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
                      'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'}
            for en, pt in months.items():
                response = response.replace(en, pt)
            self.falar(response)
        except Exception as e:
            print(f"Erro ao verificar data: {e}")
            self.falar("Desculpe, ocorreu um erro ao verificar a data.")

    def treinar_modelo_facial(self):
        global detector_facial
        try:
            if RECONHECEDOR_A_SER_USADO == "LBPH":
                reconhecedor = cv2.face.LBPHFaceRecognizer_create()
            else:
                self.falar("Algoritmo de reconhecimento não suportado.")
                return
        except AttributeError:
            self.falar("Erro fatal. O módulo de reconhecimento facial não está instalado corretamente.")
            return

        def getImagensEIds():
            pasta_principal = 'fotos'
            faces, ids = [], []
            id_map = {'allex': 1, 'caique': 2, 'diego': 3, 'lucas': 4, 'maria': 5, 'pedro': 6, 'victor': 7}
            for nome_pessoa, id_numerico in id_map.items():
                caminho_pasta_pessoa = os.path.join(pasta_principal, f"fotos_{nome_pessoa}")
                if not os.path.isdir(caminho_pasta_pessoa): continue
                for nome_arquivo in os.listdir(caminho_pasta_pessoa):
                    caminho_imagem = os.path.join(caminho_pasta_pessoa, nome_arquivo)
                    imagem = cv2.imread(caminho_imagem)
                    if imagem is None: continue
                    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
                    faces_detectadas = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.2, minNeighbors=3)
                    for (x, y, l, a) in faces_detectadas:
                        faces.append(imagem_cinza[y:y + a, x:x + l])
                        ids.append(id_numerico)
            return faces, np.array(ids)

        self.falar("Iniciando o treinamento do modelo facial.")
        faces_treino, ids_treino = getImagensEIds()
        if len(faces_treino) > 0:
            reconhecedor.train(faces_treino, ids_treino)
            reconhecedor.write(NOME_ARQUIVO_MODELO_FACIAL)
            self.falar("Treinamento concluído com sucesso!")
        else:
            self.falar("Erro. Nenhum rosto foi detectado para treinamento.")

    def iniciar_reconhecimento_facial(self):
        if not os.path.exists(NOME_ARQUIVO_MODELO_FACIAL):
            self.falar("Modelo não encontrado. Treine o modelo primeiro.")
            return
        try:
            reconhecedor = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            self.falar("Erro fatal. O módulo de reconhecimento facial não está instalado corretamente.")
            return

        reconhecedor.read(NOME_ARQUIVO_MODELO_FACIAL)
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.falar("Erro ao acessar a câmera.")
            return

        self.falar("Iniciando reconhecimento facial. Pressione 'q' para sair.")
        id_to_name = {1: 'Allex', 2: 'Caique', 3: 'Diego', 4: 'Lucas', 5: 'Maria', 6: 'Pedro', 7: 'Victor'}
        while True:
            status, imagem = camera.read()
            if not status: break
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            facesDetectadas = detector_facial.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))
            for x, y, l, a in facesDetectadas:
                rosto_recortado = cv2.resize(imagemCinza[y:y + a, x:x + l], (220, 220))
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                id_predito, confianca = reconhecedor.predict(rosto_recortado)
                nome_exibido = "Desconhecido" if confianca > 85.0 else id_to_name.get(id_predito, 'Desconhecido')
                cv2.putText(imagem, nome_exibido, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Reconhecimento Facial", imagem)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        camera.release()
        cv2.destroyAllWindows()
        self.falar("Reconhecimento facial finalizado.")

    def iniciar_ocr(self):
        self.falar("Iniciando OCR. Pressione Q para ler e sair.")
        try:
            reader = easyocr.Reader(['pt'])
        except Exception as e:
            self.falar("Erro ao inicializar o leitor OCR.")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.falar("Erro ao acessar a câmera para OCR.")
            return
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resultado = reader.readtext(gray)
            texto_reconhecido = []
            for (bbox, text, prob) in resultado:
                if prob > 0.5:
                    texto_reconhecido.append(text)
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    br = (int(br[0]), int(br[1]))
                    cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            cv2.imshow("OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                texto_final = " ".join(texto_reconhecido)
                if texto_final:
                    self.falar(f"Texto lido: {texto_final}")
                else:
                    self.falar("Nenhum texto claro foi detectado.")
                break
        cap.release()
        cv2.destroyAllWindows()
        self.falar("OCR finalizado.")

    def tirar_screenshot(self, name):
        self.falar("Capturando a tela em 3 segundos...")
        time.sleep(3)
        name_clean = name.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ',
                                                                                               '') or f'screenshot_{int(time.time())}'
        pyautogui.screenshot(f'{name_clean}.png')
        self.falar(f"Screenshot salva como {name_clean}.png")

    def previsao_wttr(self, location):
        try:
            # Correção feita: adicionado &lang=pt para forçar resposta em português
            resp = requests.get(f"https://wttr.in/{location}?format=j1&lang=pt", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            area = data["nearest_area"][0]["areaName"][0]["value"]
            current = data["current_condition"][0]
            msg = f"A previsão em {area} é de {current['temp_C']} graus, com {current['weatherDesc'][0]['value']}."
            self.falar(msg)
        except requests.RequestException:
            self.falar(f"Erro ao obter a previsão para {location}.")

    def cotacao_moeda(self, texto_comando):
        mapa_moedas = {'dólar': 'USD', 'euro': 'EUR', 'bitcoin': 'BTC', 'iene': 'JPY', 'libra': 'GBP'}
        moeda_encontrada = None
        for nome_moeda, codigo in mapa_moedas.items():
            if nome_moeda in texto_comando:
                moeda_encontrada = nome_moeda
                codigo_moeda = codigo
                break
        if not moeda_encontrada:
            self.falar("Não conheço essa moeda.")
            return
        try:
            url = f"https://economia.awesomeapi.com.br/json/last/{codigo_moeda}-BRL"
            data = requests.get(url, timeout=5).json()
            dados_moeda = data[f"{codigo_moeda}BRL"]
            valor = float(dados_moeda['bid'])
            if codigo_moeda == 'BTC':
                self.falar(f"1 {moeda_encontrada} está valendo R$ {valor:,.2f}".replace(",", "."))
            else:
                self.falar(f"A cotação do {moeda_encontrada} hoje é de R$ {valor:.2f}")
        except Exception:
            self.falar(f"Desculpe, não consegui encontrar a cotação para {moeda_encontrada}.")

    def tocar_musica_youtube(self, nome_musica):
        url = f"https://www.youtube.com/results?search_query={quote(nome_musica)}"
        webbrowser.open(url)
        self.falar(f"Procurando por {nome_musica} no YouTube.")

    # =========================================================================
    # LOOP PRINCIPAL DE COMANDOS DO ASSISTENTE
    # =========================================================================
    def _run_voice_command_loop(self):
        calculo_keywords = ['calcule', 'quanto é', 'quanto dá', 'raiz', '+', '-', 'x', '/', 'ao quadrado', 'elevado a']
        cotacao_keywords = ['cotação', 'preço', 'valor', 'quanto vale', 'dólar', 'euro', 'bitcoin', 'libra', 'iene']
        CORVO_KEYWORDS = ['ei corvo', 'rei corvo', 'ok corvo']

        while self.main_loop_running:
            try:
                print(COMANDOS_LISTA)
                with sr.Microphone() as mic:
                    self.reconhecedor.adjust_for_ambient_noise(mic, duration=1.0)
                    print("ESCUTANDO...")
                    audio = self.reconhecedor.listen(mic, timeout=5, phrase_time_limit=5)
                print("Reconhecendo...")
                texto = self.reconhecedor.recognize_google(audio, language='pt-BR')
                texto_lower = texto.lower()
                print(f"Você disse: '{texto_lower}'")

                found_keyword = None
                for keyword in CORVO_KEYWORDS:
                    if texto_lower.startswith(keyword):
                        found_keyword = keyword
                        break

                if 'parar' in texto_lower or 'sair' in texto_lower:
                    self.falar("Saindo do programa. Até logo!")
                    self.main_loop_running = False
                    break
                elif 'agenda' in texto_lower:
                    self.menu_agenda()
                elif 'iniciar ocr' in texto_lower:
                    self.iniciar_ocr()
                elif 'iniciar reconhecimento facial' in texto_lower:
                    self.iniciar_reconhecimento_facial()
                elif 'iniciar reconhecimento de animais' in texto_lower:
                    self.classificador_keras.prever_com_webcam()
                elif 'treinar facial' in texto_lower:
                    self.treinar_modelo_facial()
                elif 'classificar animais' in texto_lower:
                    self.classificador_keras.prever_de_arquivo()
                elif "screenshot" in texto_lower:
                    name = texto_lower.split("nome", 1)[-1].strip() if "nome" in texto_lower else ""
                    self.tirar_screenshot(name)
                elif 'tocar' in texto_lower and ('música' in texto_lower or 'musica' in texto_lower):
                    nome_musica = re.split(r'tocar (?:a )?m[uú]sica', texto_lower, 1)[-1].strip()
                    if nome_musica:
                        self.tocar_musica_youtube(nome_musica)
                    else:
                        self.falar("Qual música?")

                elif found_keyword:
                    user_prompt = texto_lower.replace(found_keyword, '', 1).strip()
                    if user_prompt:
                        self.perguntar_groq(user_prompt)
                    else:
                        self.falar("Sim? Qual é a sua pergunta?")

                elif any(keyword in texto_lower for keyword in cotacao_keywords):
                    self.cotacao_moeda(texto_lower)
                elif any(keyword in texto_lower for keyword in calculo_keywords) or re.search(r'\d', texto_lower):
                    self.calculadora(texto_lower)
                elif "previsão do tempo" in texto_lower:
                    cidade = texto_lower.split("em", 1)[-1].strip()
                    if cidade != "previsão do tempo":
                        self.previsao_wttr(cidade)
                    else:
                        self.falar("Qual cidade?")
                elif 'que horas são' in texto_lower:
                    self.data_hora()
                elif 'qual é a data' in texto_lower or 'que dia é hoje' in texto_lower:
                    self.data_completa()
                elif 'o que você faz' in texto_lower:
                    funcoes = "Eu posso gerenciar sua agenda, fazer cálculos, informar data e hora, realizar reconhecimento facial e de animais, ler textos, interagir com a inteligência artificial do Corvo, tirar screenshots, dar a previsão do tempo, cotações de moedas e tocar músicas."
                    self.falar(funcoes)
                else:
                    self.falar("Comando não reconhecido.")
            except sr.UnknownValueError:
                self.falar("Não entendi. Pode repetir?")
            except sr.WaitTimeoutError:
                print("Tempo de escuta excedido.")
            except Exception as erro:
                print(f"Ocorreu um erro inesperado no loop principal: {erro}")

    def assistente_principal(self):
        self.main_loop_running = True
        self.falar("Olá! Eu sou a sua assistente virtual. Diga o seu comando quando eu estiver escutando.")
        self._run_voice_command_loop()
        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    assistente = AssistenteMaster()
    assistente.assistente_principal()
