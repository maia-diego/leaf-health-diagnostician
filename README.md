# Leaf Health Diagnostician

Este projeto é uma aplicação de aprendizado de máquina para classificar a saúde das folhas de plantas. Utiliza redes neurais convolucionais (CNNs) para analisar imagens de folhas e determinar se estão saudáveis ou doentes.

## Estrutura do Projeto

- **`src/`**: Contém o código-fonte da aplicação.
  - **`train.py`**: Script para treinar o modelo de classificação.
  - **`classify.py`**: Script para classificar imagens usando um modelo pré-treinado.
  - **`data/`**: Módulo para carregamento e pré-processamento de dados.
- **`models/`**: Diretório onde os modelos treinados são salvos.
- **`images/`**: Diretório contendo conjuntos de dados de imagens de folhas.
- **`config.yaml`**: Arquivo de configuração para definir parâmetros de treinamento e classificação.
- **`history/`**: Diretório onde o histórico de treinamento é salvo.
- **`train/`**: Diretório onde gráficos de desempenho são salvos.

## Pré-requisitos

- Python 3.8 ou superior
- Anaconda ou pip para gerenciar pacotes
- PyTorch
- Torchvision
- PIL (Python Imaging Library)
- scikit-learn
- Matplotlib
- PyYAML
- Pandas

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/leaf-health-diagnostician.git
   cd leaf-health-diagnostician


Crie um ambiente virtual e ative-o:


conda create --name leaf-health python=3.8
conda activate leaf-health


Instale as dependências:


pip install -r requirements.txt


Uso
Treinamento do Modelo
Para treinar um modelo, execute o script train.py. Certifique-se de que o arquivo config.yaml está configurado corretamente para o conjunto de dados e parâmetros desejados.

python src/train.py

Classificação de Imagens
Para classificar imagens usando um modelo pré-treinado, execute o script classify.py. Atualize o caminho do modelo e o diretório de imagens no script conforme necessário.

python src/classify.py

Configuração
O arquivo config.yaml contém configurações para treinamento e classificação, incluindo diretórios de dados, hiperparâmetros de treinamento e configurações de logging. Certifique-se de ajustar esses parâmetros conforme necessário para seu uso específico.

Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.


