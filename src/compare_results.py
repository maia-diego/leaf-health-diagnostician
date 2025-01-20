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
import pandas as pd
import numpy as np
from datetime import datetime

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_comparison_directory():
    """Cria o diretório de comparação se não existir."""
    comparison_dir = 'comparison'
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
        logging.info(f"Diretório de comparação '{comparison_dir}' criado.")
    return comparison_dir

def all_measures(target, predicted):
    """Calcula e retorna várias métricas de desempenho."""
    acc = accuracy_score(target, predicted)
    prec = precision_score(target, predicted, average='weighted')
    rec = recall_score(target, predicted, average='weighted')
    f_me = f1_score(target, predicted, average='weighted')
    return acc, prec, rec, f_me

def load_model(model_path, model_type, num_classes=2):
    """Carrega o modelo especificado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == 'googlenet':
        model = models.googlenet(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'custom':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150*150*3, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    else:
        raise ValueError("Tipo de modelo não suportado")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)

def evaluate_model(model, test_loader, device):
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return all_targets, all_preds

def compare_models(config):
    """Compara os modelos GoogLeNet e Custom."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparison_dir = create_comparison_directory()
    
    # Carregar dados de teste
    _, _, test_loader = load_data(config['data']['training']['data_dir'], config['data']['training']['batch_size'])
    
    # Carregar e avaliar modelos
    models_to_evaluate = {
        'GoogLeNet': load_model('models/googlenet_leaf_health_model.pth', 'googlenet'),
        'Custom': load_model('models/custom_leaf_health_model.pth', 'custom')
    }
    
    results = {}
    for model_name, model in models_to_evaluate.items():
        targets, predictions = evaluate_model(model, test_loader, device)
        acc, prec, rec, f1 = all_measures(targets, predictions)
        results[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }
    
    # Criar e salvar gráfico de comparação
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [results['GoogLeNet'][m] for m in metrics], width, label='GoogLeNet')
    plt.bar(x + width/2, [results['Custom'][m] for m in metrics], width, label='Custom')
    
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(comparison_dir, f'{timestamp}_model_comparison.png'))
    plt.close()
    
    # Salvar resultados em um DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(comparison_dir, f'{timestamp}_model_comparison_results.csv'))
    
    logging.info(f"Comparação de modelos concluída. Resultados salvos em {comparison_dir}")

if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    compare_models(config)
