import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from src.data import load_data
import yaml
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd  # Importando pandas para salvar histórico

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    logging.info(f"Carregando configurações do arquivo: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_bias_variance(target, predicted):
    """Calcula o viés e a variância das previsões do modelo."""
    bias = np.mean(predicted) - np.mean(target)
    variance = np.var(predicted)
    logging.debug(f"Cálculo de viés: {bias}, variância: {variance}")
    return bias, variance

config = load_config()
log_file = os.path.join(config['logging']['log_dir'], f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Configure logging to save to a file with a timestamp and show in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adiciona o handler do arquivo ao logger
logging.getLogger().addHandler(file_handler)

class LeafHealthClassifier:
    """Classe para treinar um classificador de saúde das folhas."""

    def __init__(self, config):
        """Inicializa o classificador com as configurações fornecidas."""
        logging.info("Inicializando o classificador de saúde das folhas.")
        self.data_dir = config['data']['training']['data_dir']
        self.epochs = config['data']['training']['num_epochs']
        self.lr = config['data']['training']['learning_rate']
        self.batch_size = config['data']['training']['batch_size']
        self.dataset_fraction = config['data']['training'].get('dataset_fraction', 1.0)  # Padrão é 100%
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model()
        self.writer = SummaryWriter(log_dir='runs/train')
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'bias': [], 'variance': []}  # Para armazenar o histórico

    def initialize_model(self):
        """Inicializa e configura o modelo GoogLeNet."""
        logging.info("Inicializando o modelo GoogLeNet.")
        model = models.googlenet(aux_logits=False, init_weights=True)  # Adiciona init_weights=True
        model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustando para 2 classes
        model = model.to(self.device)
        self.initialize_weights(model)
        return model

    def initialize_weights(self, model):
        """Inicializa os pesos da rede neural com a técnica de He."""
        logging.info("Inicializando pesos do modelo.")
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def train(self):
        """Função principal para treinar o modelo."""
        logging.info("Iniciando o treinamento do modelo.")
        train_loader, val_loader, _ = load_data(self.data_dir, self.batch_size, self.dataset_fraction)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        best_val_loss = float('inf')
        patience = 5  # Para Early Stopping

        # Medir tempo de treinamento
        total_training_time = 0

        for epoch in range(self.epochs):
            start_time = time.time()  # Início do tempo de treinamento para a época
            self.model.train()  # Muda o modelo para modo de treinamento
            total_loss = 0
            
            logging.info(f"Iniciando a época {epoch + 1}/{self.epochs}.")
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)

                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                    loss = criterion(main_output, target)
                    if len(outputs) > 1:  # Se há saída auxiliar
                        aux_output = outputs[1]
                        aux_loss = criterion(aux_output, target)
                        loss += 0.4 * aux_loss  # Peso da perda auxiliar
                else:
                    loss = criterion(outputs, target)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Perda média de treinamento na época {epoch + 1}: {avg_train_loss:.4f}")

            # Loop de validação
            val_loss, accuracy = self.validate(val_loader, criterion)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/validation', accuracy, epoch)
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

            # Armazenar histórico
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(accuracy)
            self.history['val_accuracy'].append(accuracy)

            # Cálculo do viés e variância na validação
            bias, variance = calculate_bias_variance(target.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy())
            self.history['bias'].append(bias)
            self.history['variance'].append(variance)

            # Ajusta a taxa de aprendizado
            scheduler.step(val_loss)

            # Implementa Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(self.model.state_dict(), f"models/{timestamp}_leaf_health_model.pth")
                patience = 5
                logging.info(f"Modelo salvo com melhor perda de validação: {best_val_loss:.4f}")
            else:
                patience -= 1
                if patience == 0:
                    logging.info("Early stopping triggered.")
                    break

            # Calcular e registrar o tempo de treinamento para a época
            epoch_time = time.time() - start_time
            total_training_time += epoch_time
            logging.info(f"Tempo de treinamento para a época {epoch + 1}: {epoch_time:.2f} segundos")

        logging.info(f"Tempo total de treinamento: {total_training_time:.2f} segundos")
        self.writer.close()  # Fecha o TensorBoard writer

        # Plotar gráficos de desempenho
        self.save_metrics()

    def save_history(self, filename='training_history.csv'):
        """Salva o histórico de treinamento em um arquivo CSV com timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Formato: YYYYMMDD_HHMMSS
        history_dir = 'history'
        
        # Cria o diretório se não existir
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # Define o caminho completo do arquivo
        full_filename = os.path.join(history_dir, f'{timestamp}_{filename}')
        
        # Salva o DataFrame em um arquivo CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(full_filename, index=False)
        logging.info(f"Histórico de treinamento salvo em {full_filename}.")

    def validate(self, val_loader, criterion):
        """Função para validar o modelo.""" 
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                val_loss += criterion(outputs, target).item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        logging.info(f"Validação concluída: Val Loss: {avg_val_loss:.4f}, Acurácia: {accuracy:.4f}")
        return avg_val_loss, accuracy

    def save_metrics(self):
        """Função para salvar as métricas de desempenho em arquivos de imagem sem exibi-las.""" 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_dir = 'train'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        plt.figure(figsize=(12, 5))
        
        # Gráfico de perda
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.0, 1])
        plt.legend(loc='upper right')
        plt.title('Gráfico de Perda')
        plt.savefig(os.path.join(train_dir, f'{timestamp}_grafico_perda.png'))  # Salva a imagem com timestamp
        plt.close()

        # Gráfico de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Accuracy')
        plt.plot(self.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.title('Gráfico de Acurácia')
        plt.savefig(os.path.join(train_dir, f'{timestamp}_grafico_acuracia.png'))  # Salva a imagem com timestamp
        plt.close()

# Verifica se o arquivo está sendo executado
if __name__ == "__main__":
    config = load_config()  # Carrega as configurações do arquivo YAML
    trainer = LeafHealthClassifier(config)
    trainer.train()
    trainer.save_history()
