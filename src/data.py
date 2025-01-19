import logging
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml

# Configure logging to save to a file with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Formato: YYYYMMDD_HHMMSS
log_file = f'logs/data_log_{timestamp}.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path="config.yaml"):
    """Carrega as configurações do arquivo YAML."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, transform=None):
    """
    Carrega o conjunto de dados e o divide em conjuntos de treinamento, validação e teste.

    Args:
        data_dir (str): Caminho para o diretório do conjunto de dados.
        batch_size (int): Tamanho do lote para o DataLoader.
        train_ratio (float): Proporção do conjunto de dados a incluir no conjunto de treinamento.
        val_ratio (float): Proporção do conjunto de dados a incluir no conjunto de validação.
        transform (callable, optional): Transformações a serem aplicadas às imagens.

    Returns:
        DataLoader: DataLoaders para treinamento, validação e teste.
    """
    logging.info("Iniciando o processo de carregamento de dados...")

    # Use default transformations if none are provided
    if transform is None:
        logging.info("Definindo transformações padrão para os dados...")
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização padrão do ImageNet
        ])
    
    # Carrega o conjunto de dados
    logging.info(f"Carregando o conjunto de dados do diretório: {data_dir}")
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except Exception as e:
        logging.error(f"Erro ao carregar o conjunto de dados: {e}")
        return None, None, None
    
    logging.info(f"Número total de amostras no conjunto de dados: {len(dataset)}")
    logging.info(f"Classes encontradas: {dataset.classes}")

    # Divide o conjunto de dados em conjuntos de treinamento, validação e teste
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    logging.info(f"Dividindo o conjunto de dados: {train_size} amostras de treinamento, {val_size} amostras de validação, {test_size} amostras de teste.")
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Cria DataLoaders
    logging.info(f"Criando DataLoaders com tamanho de lote: {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info("Processo de carregamento de dados concluído com sucesso.")
    return train_loader, val_loader, test_loader

# Exemplo de uso (remova ou comente isso antes de integrar ao seu projeto principal)
# if __name__ == "__main__":
#     config = load_config()  # Carrega as configurações do arquivo YAML
#     data_dir = config['training']['data_dir']  # Obtém o diretório de dados do arquivo de configuração
#     batch_size = config['training']['batch_size']  # Obtém o tamanho do lote do arquivo de configuração
#     train_loader, val_loader, test_loader = load_data(data_dir=data_dir, batch_size=batch_size)
#     logging.info("Train loader, validation loader, and test loader are ready.")
