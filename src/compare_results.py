import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from data import load_data

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_comparison_directory():
    """Cria o diretório de comparação se não existir."""
    comparison_dir = '/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/evaluation'
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
        logging.info(f"Diretório de comparação '{comparison_dir}' criado.")
    return comparison_dir

def infer_model_architecture(state_dict, input_shape, num_classes=2):
    """Infere a arquitetura do modelo a partir do state_dict."""
    layers = []
    current_shape = input_shape
    in_channels = input_shape[0]

    for key, value in state_dict.items():
        if 'weight' in key:
            if len(value.shape) == 4:  # Convolutional layer
                out_channels, expected_in_channels, _, _ = value.shape
                if in_channels != expected_in_channels:
                    raise ValueError(f"Esperado {expected_in_channels} canais, mas recebeu {in_channels}.")
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = out_channels
                current_shape = (current_shape[1] // 2, current_shape[2] // 2, out_channels)
            elif len(value.shape) == 2:  # Linear layer
                out_features, in_features = value.shape
                if len(current_shape) == 3:  # Flatten if necessary
                    layers.append(nn.Flatten())
                    current_shape = (np.prod(current_shape),)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.5))

    layers.append(nn.Linear(out_features, num_classes))
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

def load_model(model_path, input_shape, num_classes=2):
    """Carrega o modelo especificado e infere sua arquitetura."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar o state_dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Inferir a arquitetura do modelo
    model = infer_model_architecture(state_dict, input_shape, num_classes)
    
    # Ajustar o state_dict para corresponder ao modelo
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    # Filtrar chaves que estão no state_dict mas não no modelo
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    try:
        model.load_state_dict(filtered_state_dict, strict=False)
    except RuntimeError as e:
        logging.error(f"Erro ao carregar o state_dict: {e}")
        raise e

    return model.to(device)

def all_measures(target, predicted):
    """Calcula e retorna várias métricas de desempenho."""
    acc = accuracy_score(target, predicted)
    prec = precision_score(target, predicted, average='weighted', zero_division=0)
    rec = recall_score(target, predicted, average='weighted', zero_division=0)
    f_me = f1_score(target, predicted, average='weighted', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(target, predicted).ravel()
    specificity = tn / (tn + fp)
    return acc, prec, rec, f_me, specificity

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

def plot_metrics(results, comparison_dir, timestamp):
    """Gera gráficos de comparação de métricas."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    plt.figure(figsize=(12, 7))
    x = np.arange(len(metrics))
    width = 0.2

    for i, (model_name, metrics_values) in enumerate(results.items()):
        plt.bar(x + i*width, [metrics_values[m] for m in metrics], width, label=model_name)

    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.savefig(os.path.join(comparison_dir, f'{timestamp}_model_comparison.png'))
    plt.close()

def plot_loss(history_files, comparison_dir, timestamp):
    """Gera gráficos de perda para cada modelo."""
    plt.figure(figsize=(12, 7))
    for history_file in history_files:
        history = pd.read_csv(history_file)
        model_name = os.path.basename(history_file).split('_')[0]
        plt.plot(history['loss'], label=f'{model_name} Loss')
        plt.plot(history['val_loss'], label=f'{model_name} Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(comparison_dir, f'{timestamp}_loss_comparison.png'))
    plt.close()

def plot_confusion_matrix(targets, predictions, class_names, comparison_dir, model_name, timestamp):
    """Gera matriz de confusão."""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(comparison_dir, f'{timestamp}_{model_name}_confusion_matrix.png'))
    plt.close()

def get_latest_file(directory, extension):
    """Retorna o arquivo mais recente em um diretório com a extensão especificada."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def compare_models(config, folder):
    """Compara os modelos de um diretório."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparison_dir = create_comparison_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}
    model_path = get_latest_file(folder, '.pth')
    history_file = get_latest_file(folder, '.csv')
    
    # Carregar dados de teste
    _, _, test_loader = load_data(config['data']['training']['data_dir'], config['data']['training']['batch_size'])

    # Obter o tamanho de entrada do primeiro lote de dados
    sample_data, _ = next(iter(test_loader))
    input_shape = sample_data.shape[1:]  # Ignorar batch size

    # Carregar e avaliar o modelo
    model = load_model(model_path, input_shape)
    targets, predictions = evaluate_model(model, test_loader, device)
    acc, prec, rec, f1, spec = all_measures(targets, predictions)
    
    model_name = os.path.basename(folder)
    results[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Specificity': spec
    }

    # Plotar matriz de confusão
    plot_confusion_matrix(targets, predictions, ['Healthy', 'Diseased'], comparison_dir, model_name, timestamp)

    # Plotar gráficos de comparação de métricas
    plot_metrics(results, comparison_dir, timestamp)

    # Plotar gráfico de perda
    plot_loss([history_file], comparison_dir, timestamp)

    # Salvar resultados em um DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(comparison_dir, f'{timestamp}_model_comparison_results.csv'))
    
    logging.info(f"Comparação de modelos concluída para {model_name}. Resultados salvos em {comparison_dir}")

if __name__ == "__main__":
    config = load_config()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Processar cada pasta individualmente
    folders = [
        '/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/analises/patience_blocked',
        '/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/analises/patient_blocked2',
        '/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/analises/arch3',
        '/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/analises/googleNet',
    ]

    for folder in folders:
        try:
            compare_models(config, folder)
        except ValueError as e:
            logging.error(f"Erro ao processar {folder}: {e}")
