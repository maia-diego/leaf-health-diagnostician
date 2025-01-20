import sys
import os
import yaml
import logging
import argparse
from datetime import datetime

# Ajustando o caminho do sistema para importações locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importações das funções
try:
    from src.train import LeafHealthClassifier  # Atualizado para importar do novo train.py
    from src.evaluate import evaluate_model
    from src.explain import explain_model
    from src.classify_googleNet import classify_images 
    logging.info("Módulos importados com sucesso.")
except ImportError as e:
    logging.error(f"Erro ao importar módulos: {e}")

# Carregar configuração
def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    logging.info(f"Carregando configurações do arquivo: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Carregar configuração
config = load_config()
image_dir = config['data']['evaluation']['image_dir']  # Corrigido para acessar a chave corretamente

log_file = os.path.join(config['logging']['log_dir'], f'main_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

# Configure logging to save to a file with a timestamp and show in terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adiciona o handler do arquivo ao logger
logging.getLogger().addHandler(file_handler)

logging.info(f"Diretório de imagens para avaliação: {image_dir}")

def main():
    parser = argparse.ArgumentParser(description="Leaf Health Diagnostician")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo")
    parser.add_argument("--evaluate", action="store_true", help="Avaliar o modelo")
    parser.add_argument("--explain", action="store_true", help="Explicar as predições")
    parser.add_argument("--classify", type=str, help="Caminho para o diretório contendo imagens para classificação")
    
    args = parser.parse_args()
    logging.info(f"Argumentos fornecidos: {args}")

    if args.train:
        logging.info('Iniciando o treinamento...')
        trainer = LeafHealthClassifier(config)  # Criar uma instância da classe
        trainer.train()  # Chamar o método de treinamento
        trainer.save_history()
        logging.info('Treinamento concluído e histórico salvo.')
    elif args.evaluate:
        logging.info('Iniciando avaliação do modelo...')
        evaluate_model(data_dir=config['data']['training']['data_dir'], batch_size=config['data']['training']['batch_size'])  # Passa o diretório de dados e o tamanho do lote
        logging.info('Avaliação do modelo concluída.')
    elif args.explain:
        logging.info('Iniciando explicação das predições...')
        explain_model(data_dir=config['data']['training']['data_dir'])
        logging.info('Explicação das predições concluída.')
    elif args.classify:
        logging.info(f'Classificando imagens no diretório: {args.classify}')
        model_path = config['model']['save_path']  # Obtém o caminho do modelo do arquivo de configuração
        class_names = ["Diseased", "Healthy"]  # Nomes das classes correspondentes às saídas do modelo
        classify_images(image_dir=args.classify, model_path=model_path, class_names=class_names)
        logging.info('Classificação de imagens concluída.')
    else:
        # Se --classify não for fornecido, use o diretório do arquivo de configuração
        logging.info(f'Classificando imagens no diretório padrão: {image_dir}')
        model_path = config['model']['save_path']  # Obtém o caminho do modelo do arquivo de configuração
        class_names = ["Healthy", "Diseased"]  # Nomes das classes correspondentes às saídas do modelo  
        classify_images(image_dir=image_dir, model_path=model_path, class_names=class_names)
        logging.info('Classificação de imagens concluída no diretório padrão.')

if __name__ == "__main__":
    logging.info("Iniciando o script Leaf Health Diagnostician.")
    main()
    logging.info("Script concluído.")
