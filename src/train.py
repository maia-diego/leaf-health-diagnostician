import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.data import load_data
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_weights(model):
    """Inicializa os pesos da rede neural com a técnica de He."""
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

def train_model(data_dir="data", epochs=20, lr=0.001, batch_size=32):
    """Função de treinamento do modelo."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = load_data(data_dir, batch_size)

    # Inicializa e configura o modelo
    model = models.googlenet(aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajusta para classificação binária
    model = model.to(device)

    # Inicializando os pesos
    initialize_weights(model)

    # Define a função de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Regularização L2

    # Scheduler para a taxa de aprendizado
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    # TensorBoard para monitoramento
    writer = SummaryWriter(log_dir='runs/train')

    # Inicialização de variáveis para Early Stopping
    best_val_loss = float('inf')
    patience = 5  # Para Early Stopping

    for epoch in range(epochs):
        model.train()  # Muda o modelo para modo de treinamento
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Se a saída é um tupla (GoogLeNetOutputs), extraia a saída principal
            if isinstance(outputs, tuple):  # No caso do GoogLeNet
                main_output = outputs[0]  # A primeira saída é a principal
                loss = criterion(main_output, target)  # Calcula a perda usando a saída principal
                if len(outputs) > 1:  # Se há saída auxiliar
                    aux_output = outputs[1]
                    aux_loss = criterion(aux_output, target)
                    loss += 0.4 * aux_loss  # Peso da perda auxiliar
            else:
                loss = criterion(outputs, target)  # No caso que não seja GoogLeNet
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        

        # Loop de validação
        model.eval()  # Muda o modelo para modo de avaliação
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

        # Ajusta a taxa de aprendizado
        scheduler.step(avg_val_loss)

        # Implementa Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Salva o melhor modelo
            torch.save(model.state_dict(), "models/leaf_health_model.pth")
            logging.info("Melhor modelo salvo.")
            patience = 5  # Reinicia a paciência
        else:
            patience -= 1
            if patience == 0:
                logging.info("Early stopping triggered.")
                break

    writer.close()  # Fecha o TensorBoard writer

# Verifica se o arquivo está sendo executado
if __name__ == "__main__":
    train_model(data_dir='data', epochs=20, lr=0.001, batch_size=32)
