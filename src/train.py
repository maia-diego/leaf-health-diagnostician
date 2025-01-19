import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import yaml
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # Importando SummaryWriter

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    logging.info(f"Carregando configurações do arquivo: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class CustomNeuralNet(nn.Module):
    """Classe para definir uma arquitetura de rede neural personalizada."""

    def __init__(self):
        """Inicializa a rede neural e suas camadas."""
        super(CustomNeuralNet, self).__init__()
        logging.info("Inicializando a arquitetura da rede neural personalizada.")

        # Definindo as camadas
        self.fc1 = nn.Linear(150*150*3, 512)  # Camada de entrada
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # 50% de dropout

        self.fc2 = nn.Linear(512, 256)  # Camada intermediária
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)  # 50% de dropout

        self.fc3 = nn.Linear(256, 2)  # Camada de saída
        self.softmax = nn.Softmax(dim=1)  # Softmax para a camada de saída

    def forward(self, x):
        """Define a passagem para frente da rede."""
        x = x.view(-1, 150*150*3)  # Aplanar a entrada
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

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
        self.model = CustomNeuralNet().to(self.device)  # Usando a nova rede neural personalizada
        self.writer = SummaryWriter(log_dir='runs/train')
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'bias': [], 'variance': []}  # Para armazenar o histórico

    def train(self):
        """Função principal para treinar o modelo."""
        logging.info("Iniciando o treinamento do modelo.")
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])
        
        train_data = ImageFolder(root=self.data_dir, transform=transform)
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        val_data = ImageFolder(root=self.data_dir.replace('training', 'validation'), transform=transform)
        val_loader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience = 5  # Para Early Stopping

        for epoch in range(self.epochs):
            start_time = time.time()  # Início do tempo de treinamento para a época
            self.model.train()  # Muda o modelo para modo de treinamento
            total_loss = 0
            
            logging.info(f"Iniciando a época {epoch + 1}/{self.epochs}.")
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
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

            # Armazenar histórico
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(accuracy)
            self.history['val_accuracy'].append(accuracy)

            # Ajusta a taxa de aprendizado
            # Aqui pode ser adicionado um scheduler se desejado

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
            logging.info(f"Tempo de treinamento para a época {epoch + 1}: {epoch_time:.2f} segundos")

        self.writer.close()  # Fecha o TensorBoard writer

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
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
        logging.info(f"Validação concluída: Val Loss: {avg_val_loss:.4f}, Acurácia: {accuracy:.4f}")
        return avg_val_loss, accuracy

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

# Configurações de logging
log_file = os.path.join(config['logging']['log_dir'], f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = load_config()  # Carrega as configurações do arquivo YAML
    
    # Inicialização do classificador
    trainer = LeafHealthClassifier(config)
    trainer.train()
    trainer.save_history()

if __name__ == "__main__":
    main()
