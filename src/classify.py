import os
import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
from src.data import load_data

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_model(model_path):
    """Carrega o modelo pré-treinado a partir do caminho especificado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(aux_logits=True)  # Se você estiver usando a versão sem saídas auxiliares
    model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustando para 2 classes
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        raise e  # Propagar a exceção após registrar o erro

    model.eval()  # Coloca o modelo em modo de avaliação
    return model

def classify_images(image_dir):
    """Classifica as imagens em um diretório especificado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Setting up device for inference.")
    
    model_path = "models/leaf_health_model.pth"
    logging.info(f"Loading model from {model_path}.")
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
        # Verifica se o arquivo é uma imagem
        img_path = os.path.join(image_dir, img_name)
        
        try:
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)  # Adiciona a dimensão do batch e move para o dispositivo

            # Forward pass
            with torch.no_grad():
                output = model(image)
                prediction = output.argmax(dim=1).item()
                results.append((img_name, prediction))
                
            logging.info(f"Classified {img_name} as class {prediction}.")
        
        except Exception as e:
            logging.error(f"Error loading image {img_name}: {e}")

    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify images in a directory.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images for classification")
    args = parser.parse_args()
    
    classify_images(image_dir=args.image_dir)
