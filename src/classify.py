import os
from datetime import datetime
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import yaml

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(model_path):
    """Carrega o modelo pré-treinado a partir do caminho especificado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(aux_logits=False)  # Desabilitando a saída auxiliar
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustando para 2 classes
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        raise e  # Propagar a exceção após registrar o erro

    model.eval()  # Coloca o modelo em modo de avaliação
    return model

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
                
            logging.info(f"Classificado {img_name} como classe '{prediction_name}'.")
        
        except Exception as e:
            logging.error(f"Erro ao carregar a imagem {img_name}: {e}")

    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify images in a directory.")
    parser.add_argument("image_dir", type=str, help="Caminho para o diretório contendo imagens para classificação")
    args = parser.parse_args()
    
    config = load_config()  # Carrega as configurações do arquivo YAML
    model_path = config['model']['save_path']  # Obtém o caminho do modelo do arquivo de configuração

    # Definindo nomes das classes
    class_names = ["Healthy", "Diseased"]  # Nomes das classes correspondentes às saídas do modelo

    classify_images(image_dir=args.image_dir, model_path=model_path, class_names=class_names)
