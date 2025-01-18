import sys
import os
import yaml
import logging
import argparse

# Ajustando o caminho do sistema para importações locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuração de log
logging.basicConfig(level=logging.INFO)

# Importações das funções
try:
    from src.train import train_model
    from src.evaluate import evaluate_model
    from src.explain import explain_model
    from src.classify import classify_images 
    logging.info("Módulos importados com sucesso.")
except ImportError as e:
    logging.error(f"Erro ao importar módulos: {e}")

# Carregar configuração
def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

num_epochs = config['training']['num_epochs']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
data_dir = config['training']['data_dir']  # Caminho do diretório de dados

def main():
    parser = argparse.ArgumentParser(description="Leaf Health Diagnostician")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo")
    parser.add_argument("--evaluate", action="store_true", help="Avaliar o modelo")
    parser.add_argument("--explain", action="store_true", help="Explicar as predições")
    parser.add_argument("--classify", type=str, help="Caminho para o diretório contendo imagens para classificação")
    
    args = parser.parse_args()

    if args.train:
        logging.info('Iniciando o treinamento...')
        train_model(data_dir=data_dir, epochs=num_epochs, lr=learning_rate, batch_size=batch_size)
    elif args.evaluate:
        logging.info('Iniciando avaliação do modelo...')
        evaluate_model(data_dir=data_dir)  # Passa o diretório de dados
    elif args.explain:
        logging.info('Iniciando explicação das predições...')
        explain_model()
    elif args.classify:
        logging.info(f'Classificando imagens no diretório: {args.classify}')
        classify_images(image_dir=args.classify)
    else:
        logging.error("Uso incorreto. Utilize --train, --evaluate, --explain ou --classify <caminho_das_imagens>.")

if __name__ == "__main__":
    main()
