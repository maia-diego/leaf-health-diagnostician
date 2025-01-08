# **Leaf Health Diagnostician**  
**Classificador de Folhas de Manga: Saudáveis vs Doentes**

---

## **Descrição do Projeto**
O **Leaf Health Diagnostician** é um sistema de classificação binária desenvolvido para identificar automaticamente o estado de saúde de folhas de manga. Ele utiliza redes neurais convolucionais (CNNs) para analisar imagens de alta resolução e categorizar as folhas como **saudáveis** ou **doentes**. Este projeto busca contribuir para o monitoramento agrícola, permitindo a detecção precoce de doenças em plantações de manga, melhorando a produtividade e reduzindo perdas.

---

## **Sobre o Dataset**
O projeto utiliza o **Mango Leaf Dataset: Healthy vs Diseased**, que contém imagens de alta resolução organizadas em duas classes:

### **Classes**
1. **Saudável:** Folhas sem sinais visíveis de doenças.
2. **Doente:** Folhas com sintomas como descoloração ou manchas.

### **Detalhes do Dataset**
- **Balanceamento de Classes:** Aproximadamente 50% de folhas saudáveis e 50% de folhas doentes.
- **Formato:** Imagens de alta resolução padronizadas.
- **Licenciamento:** Disponível sob a licença Attribution 4.0 International, permitindo acesso aberto com atribuição adequada.
- **Fonte:** O dataset está disponível nas plataformas [Zenodo](https://zenodo.org) e [Kaggle](https://kaggle.com).

### **Aplicações**
- **Aprendizado de Máquina:** Treinamento de modelos de classificação binária.
- **Monitoramento de Doenças:** Desenvolvimento de sistemas de diagnóstico para a saúde de plantações de manga.
- **Processamento de Imagens:** Pesquisa em extração de características e segmentação.

---

## **Objetivo do Projeto**
- **Automação no Diagnóstico:** Auxiliar agricultores e pesquisadores a identificar rapidamente doenças em folhas de manga.
- **Desenvolvimento Sustentável:** Reduzir perdas agrícolas por meio da detecção precoce de doenças.
- **Explicabilidade:** Incorporar técnicas como SHAP (SHapley Additive ExPlanations) para explicar as decisões do modelo.

---

## **Estrutura do Projeto**
- **Modelos:** Treinamento de redes convolucionais (CNNs) utilizando o PyTorch.
- **Métricas:** Avaliação com relatórios de classificação, matriz de confusão e análise de viés e variância.
- **Explicabilidade:** Utilização do SHAP para interpretação das predições do modelo.
- **Dataset:** Armazenado e processado localmente, com transformações de normalização e redimensionamento aplicadas.

---

## **Como Executar**
### **Pré-requisitos**
- Python 3.8+ instalado
- Dependências listadas no arquivo `requirements.txt`

### **Passos**
1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/maia-diego/leaf-health-diagnostician.git
   cd leaf-health-diagnostician
