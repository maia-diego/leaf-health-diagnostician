import os
import numpy as np
from datetime import datetime
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import yaml
import time

# Função para carregar as configurações do arquivo YAML
def load_config(file_path="config.yaml"):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Função para inferir a arquitetura do modelo a partir do state_dict
def infer_model_architecture(state_dict, num_classes=2):
    layers = []
    in_channels = 3  # Número de canais de entrada (RGB)
    input_size = 128  # Supondo um tamanho de entrada de 128x128

    for key, value in state_dict.items():
        if 'weight' in key:
            if len(value.shape) == 4:  # Camada convolucional
                out_channels, expected_in_channels, _, _ = value.shape
                if in_channels != expected_in_channels:
                    raise ValueError(f"Esperado {expected_in_channels} canais, mas recebeu {in_channels}.")
                
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = out_channels
                input_size //= 2  # Reduz o tamanho da entrada após o max pooling
            elif len(value.shape) == 2:  # Camada linear
                out_features, in_features = value.shape
                # Achata a entrada para corresponder ao tamanho esperado
                layers.append(nn.Flatten())
                layers.append(nn.Linear(in_channels * input_size * input_size, out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.5))

    layers.append(nn.Linear(out_features, num_classes))
    return nn.Sequential(*layers)

# Função para carregar o modelo
def load_model(model_path, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model = infer_model_architecture(state_dict, num_classes)
    
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    try:
        model.load_state_dict(filtered_state_dict, strict=False)
    except RuntimeError as e:
        logging.error(f"Erro ao carregar o state_dict: {e}")
        raise e

    return model.to(device)

def classify_images(image_dir, model_path, class_names):
    """Classifica as imagens em um diretório especificado.

    Args:
        image_dir (str): Caminho para o diretório contendo as imagens para classificação.
        model_path (str): Caminho para o modelo pré-treinado.
        class_names (list): Lista de nomes das classes.

    Returns:
        list: Lista de tuplas contendo o nome da imagem e a predição.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Configurando dispositivo para inferência.")
    
    logging.info(f"Carregando modelo de {model_path}.")
    model = load_model(model_path)

    # Definindo transformação das imagens
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    results = []
    class_counts = {class_name: 0 for class_name in class_names}

    start_time = time.time()  # Inicia a contagem do tempo

    # Para cada imagem no diretório,
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")  # Garante que a imagem esteja em RGB
            image = transform(image).unsqueeze(0).to(device)  # Adiciona a dimensão do batch e move para o dispositivo

            # Forward pass
            with torch.no_grad():
                output = model(image)
                prediction_index = output.argmax(dim=1).item()
                prediction_name = class_names[prediction_index]  # Obtém o nome da classe
                results.append((img_name, prediction_name))
                class_counts[prediction_name] += 1
                
            logging.info(f"Classificado {img_name} como classe '{prediction_name}'.")
        
        except Exception as e:
            logging.error(f"Erro ao carregar a imagem {img_name}: {e}")

    total_time = time.time() - start_time  # Calcula o tempo total gasto
    logging.info(f"Tempo total gasto na classificação: {total_time:.2f} segundos")

    total_images = sum(class_counts.values())
    logging.info(f"Total de imagens analisadas: {total_images}")

    for class_name, count in class_counts.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        logging.info(f"Classe '{class_name}': {count} imagens ({percentage:.2f}%)")

    return results

if __name__ == "__main__":
    config = load_config()
    log_file = os.path.join(config['logging']['log_dir'], f'classify_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # Configure logging to save to a file with a timestamp and show in terminal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Adiciona o handler do arquivo ao logger
    logging.getLogger().addHandler(file_handler)

    model_path = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/analises/patient_blocked2/20250119_212343_leaf_health_model.pth'
    # image_dir = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/images/Mango_Leaf_Dataset/Healthy'
    # image_dir = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/images/Mango_Leaf_Dataset/Diseased'
    # image_dir = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/images/tomato_dataset/valid/Tomato___healthy'
    # image_dir = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/images/tomato_dataset/valid/Tomato___Late_blight'
    # image_dir = r'/home/diegomaia/workspace/dev/mestrado/leaf-health-diagnostician/images/tomato_dataset/valid/Tomato___Leaf_Mold'
    class_names = ["Diseased", "Healthy"]  # Nomes das classes correspondentes às saídas do modelo

    classify_images(image_dir=image_dir, model_path=model_path, class_names=class_names)
