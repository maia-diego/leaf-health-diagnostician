import os
import logging
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data import load_data

# Configuração de logging
logging.basicConfig(level=logging.INFO)

def explain_model(data_dir="data", batch_size=32):
    """Função para explicar as predições do modelo."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar os dados
    logging.info("Carregando dados para explicação...")
    _, _, test_loader = load_data(data_dir, batch_size)
    logging.info("Dados carregados com sucesso.")

    # Inicializando o modelo
    model = models.googlenet(aux_logits=True)  # Desabilitar auxiliar
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustar para 2 classes
    model = model.to(device)

    # Carregar o estado do modelo
    model_path = "models/leaf_health_model.pth"
    logging.info("Carregando o modelo...")
    try:
        # Load state_dict usando weights_only=True
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # Filtrar chaves indesejadas do estado
        if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
            del state_dict['fc.weight']
            del state_dict['fc.bias']

        # Carregar pesos no modelo
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        return

    model.eval()  # Coloque o modelo em modo de avaliação

    # Inicializando variáveis para a explicação
    explanations = []
    with torch.no_grad():  # Desativando gradientes
        for data, _ in test_loader:  # Supondo que os dados possuem targets, mas não são necessários para explicações
            data = data.to(device)

            # Forward pass
            outputs = model(data)
            predictions = outputs.argmax(dim=1)

            explanations.append(predictions.cpu().numpy())  # Armazena as previsões

    # Transformar explicações em um formato de saída desejado
    explanations = np.concatenate(explanations)
    logging.info("Explicações geradas com sucesso.")
    
    # Exibir ou salvar as explicações conforme sua necessidade
    print("Predições:", explanations)

if __name__ == "__main__":
    explain_model(data_dir='data', batch_size=32)
