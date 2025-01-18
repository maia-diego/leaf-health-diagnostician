import os
import logging
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.data import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(data_dir="data", batch_size=32):
    """Função para avaliar o modelo no conjunto de dados de teste."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_data(data_dir, batch_size)

    # Define o modelo com saídas auxiliares desativadas
    model = models.googlenet(aux_logits=True)  # Desabilitando as saídas auxiliares
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustando para 2 classes (saudável e doente)
    model = model.to(device)

    logging.info("Inicializando o modelo...")
    
    # Carregar estado do modelo
    try:
        state_dict = torch.load("models/leaf_health_model.pth", map_location=device, weights_only=True)

        # Filtrar os parâmetros para remover a camada final, se necessário
        if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            
        # Carregar os pesos no modelo
        model.load_state_dict(state_dict, strict=False)
        
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

if __name__ == "__main__":
    evaluate_model(data_dir='data', batch_size=32)
