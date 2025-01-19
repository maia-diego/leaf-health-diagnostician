import os
import logging
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import load_data
import yaml
import pandas as pd  # Para carregar o histórico
import numpy as np
from datetime import datetime

# Carregar configuração
def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    print(f"Carregando configurações do arquivo: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
log_file = os.path.join(config['logging']['log_dir'], f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Configure logging to save to a file with a timestamp and show in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adiciona o handler do arquivo ao logger
logging.getLogger().addHandler(file_handler)
# Função para garantir que o diretório exista
def create_evaluation_directory():
    evaluation_dir = 'evaluation'
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
        logging.info(f"Diretório de avaliação '{evaluation_dir}' criado.")
    return evaluation_dir

def all_measures(target, predicted):
    """Calcula e retorna várias métricas de desempenho."""
    acc = accuracy_score(target, predicted)
    prec = precision_score(target, predicted)
    rec = recall_score(target, predicted)
    f_me = f1_score(target, predicted)
    spec = specificity(target, predicted)
    logging.debug(f"Métricas calculadas - Acurácia: {acc}, Precisão: {prec}, Recall: {rec}, F1 Score: {f_me}, Especificidade: {spec}")
    return acc, prec, rec, f_me, spec

def specificity(target, predicted):
    """Calcula a especificidade."""
    tn = np.sum((target == 0) & (predicted == 0))
    fp = np.sum((target == 0) & (predicted == 1))
    
    # Verifica se a soma é zero para evitar divisão por zero
    if (tn + fp) == 0:
        return 0.0  # Retorna 0.0 ou um valor apropriado se não houver negativos
    return tn / (tn + fp)

def load_training_history():
    """Carrega o último histórico de treinamento a partir de um arquivo CSV no diretório 'history'."""
    history_dir = 'history'
    
    # Verifica se o diretório existe
    if os.path.exists(history_dir):
        # Lista todos os arquivos CSV no diretório
        csv_files = [f for f in os.listdir(history_dir) if f.endswith('.csv')]
        
        if csv_files:
            # Ordena os arquivos por data (assumindo que o nome do arquivo contém um timestamp)
            csv_files.sort(reverse=True)  # Ordena do mais recente para o mais antigo
            latest_file = os.path.join(history_dir, csv_files[0])  # Pega o mais recente
            
            history = pd.read_csv(latest_file)
            logging.info(f"Histórico de treinamento carregado de {latest_file}.")
            return history
        else:
            logging.error("Nenhum arquivo CSV encontrado no diretório 'history'.")
            return None
    else:
        logging.error(f"O diretório {history_dir} não foi encontrado.")
        return None

def calculate_bias_variance(target, predicted):
    """Calcula o viés e a variância das previsões do modelo."""
    bias = np.mean(predicted) - np.mean(target)
    variance = np.var(predicted)
    logging.debug(f"Cálculo de viés: {bias}, variância: {variance}")
    return bias, variance

def evaluate_model(data_dir, batch_size):
    """Função para avaliar o modelo no conjunto de dados de teste."""    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Carregando dados de teste...")
    _, _, test_loader = load_data(data_dir, batch_size)

    # Define o modelo com saídas auxiliares desativadas
    model = models.googlenet(aux_logits=False, init_weights=True)  # Adicionando init_weights=True
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustando para 2 classes (saudável e doente)
    model = model.to(device)

    logging.info("Inicializando o modelo...")

    # Carregar estado do modelo
    try:
        model.load_state_dict(torch.load("models/leaf_health_model.pth", map_location=device, weights_only=True))  # Adiciona weights_only=True
        logging.info("Modelo carregado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        return

    model.eval()  # Coloca o modelo em modo de avaliação

    all_preds = []
    all_targets = []

    with torch.no_grad():  # Desativando o cálculo de gradientes
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Cálculo da precisão
    accuracy = accuracy_score(all_targets, all_preds)
    logging.info(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

    # Cálculo da matriz de confusão
    matriz = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = matriz.ravel()
    logging.info(f"Matriz de confusão - tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

    # Calcular viés e variância
    bias, variance = calculate_bias_variance(all_targets, all_preds)
    logging.info(f"Viés: {bias:.4f}, Variância: {variance:.4f}")

    # Salvar a matriz de confusão como imagem
    evaluation_dir = create_evaluation_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=['Saudável', 'Doente'], yticklabels=['Saudável', 'Doente'])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(os.path.join(evaluation_dir, f'{timestamp}_matriz_confusao.png'))  # Salva a imagem com timestamp
    plt.close()  # Fecha a figura para liberar memória
    logging.info(f"Matriz de confusão salva em {os.path.join(evaluation_dir, f'{timestamp}_matriz_confusao.png')}")

    # Cálculo e exibição das métricas
    acc, prec, rec, f_me, spec = all_measures(all_targets, all_preds)
    logging.info(f"Acurácia: {acc:.2f}, Precisão: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f_me:.2f}, Especificidade: {spec:.2f}")

    # Carregar o histórico de treinamento
    history = load_training_history()
    if history is not None:
        # Salvar gráficos de desempenho como imagens
        plt.figure(figsize=(12, 5))

        # Gráfico de perda
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.0, 1])
        plt.legend(loc='upper right')
        plt.title('Gráfico de Perda')
        plt.savefig(os.path.join(evaluation_dir, f'{timestamp}_grafico_perda.png'))  # Salva a imagem com timestamp
        plt.close()

        # Gráfico de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.title('Gráfico de Acurácia')
        plt.savefig(os.path.join(evaluation_dir, f'{timestamp}_grafico_acuracia.png'))  # Salva a imagem com timestamp
        plt.close()
        logging.info(f"Gráficos de desempenho salvos em {evaluation_dir}.")

if __name__ == "__main__":
    config = load_config()  # Carrega as configurações do arquivo YAML
    data_dir = config['data']['training']['data_dir']  # Obtém o diretório de dados do arquivo de configuração
    batch_size = config['data']['training']['batch_size']  # Obtém o tamanho do lote do arquivo de configuração
    logging.info("Iniciando a avaliação do modelo...")
    evaluate_model(data_dir=data_dir, batch_size=batch_size)
    logging.info("Avaliação concluída.")
