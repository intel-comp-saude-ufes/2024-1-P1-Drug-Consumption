# :pill: Drug Consumption Classifier

Com este projeto, nós procuramos encontrar relações entre usuários de substâncias e algumas características pessoais e comportamentais.

Este repositório contém o pré-processamento, as análises e alguns testes de classificadores no dataset.

> Você pode encontrar um vídeo resumindo o projeto [aqui](https://www.youtube.com/watch?v=QMbaKQqQSow).
>
> O artigo completo sobre o projeto está disponível [aqui](report/2024_1_P1_Drug_Consumption.pdf).

## Sumário
- [Dataset](#dataset)
- [Executando os Experimentos](#executando-os-experimentos)
- [Uso](#uso)
  - [Observação](#observação)
  - [Starter Notebook](#starter-notebook)
  - [Pré-processamento](#pré-processamento)
  - [Classificação](#classificação)
  - [Classificação +](#classificação-plus)

<div id="dataset"></div>

## :open_file_folder: Dataset

O dataset possui 12 colunas de características, sendo 5 colunas categóricas e 7 colunas numéricas. As 5 colunas categóricas referem-se a características demográficas, como faixa etária e gênero. As 7 colunas numéricas são indicadores comportamentais e de personalidade.

Além disso, o dataset inclui 18 colunas adicionais relacionadas ao uso de drogas, que podem ser utilizadas como target para problemas de classificação.

O dataset utilizado neste projeto pode ser encontrado [aqui](https://www.kaggle.com/datasets/mexwell/drug-consumption-classification).

<div id="executando-os-experimentos"></div>

## :computer: Executando os Experimentos

Primeiramente, clone este repositório:

```bash
git clone https://github.com/intel-comp-saude-ufes/2024-1-P1-Drug-Consumption.git
cd 2024-1-P1-Drug-Consumption
```

Para executar os notebooks, é necessário instalar alguns pacotes. Recomendamos o uso de um ambiente virtual de sua preferência. Neste guia, utilizaremos conda, que pode ser instalado [aqui](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Certifique-se de utilizar o Python 3.10. Caso opte pelo conda, siga os comandos abaixo:

```bash
conda create --name venv -c conda-forge python=3.10
conda activate venv
```

Depois de ativar o ambiente virtual, instale as dependências do projeto com o comando:

```bash
pip install -r requirements.txt
```

<div id="uso"></div>

## Uso

<div id="observação"></div>

### Observação

> Como uma parte dos experimentos envolve uma busca de hiperparâmetros, executar o notebook inteiro pode tomar muito tempo.
>
> Assim, nos notebooks de classificação há variáveis `run` que indicam se os experimentos devem ser executados ou se devem ser carregados os dados da execução salva em `results/`.
>
> Além disso, se o intuito for somente alterar as visualizações para tornar os resultados mais interpretáveis, são disponibilizados arquivos `.csv` para cada experimento executado. Tais arquivos podem ser encontrados em `results/`.

<div id="starter-notebook"></div>

### Starter Notebook

O dataset apresenta dados pouco legíveis e de difícil interpretação direta. Para facilitar o processo de análise, o autor do dataset disponibilizou um [notebook inicial (starter notebook)](notebooks/00-starter_notebook.ipynb) que realiza a transformação desses dados, permitindo que as análises subsequentes sejam mais diretas. Realizamos algumas modificações que julgamos necessárias.

Esse notebook tem como entrada o arquivo [drug_consumption](data/drug_consumption.csv) e gera como saída o arquivo [drug_consumption_prepared](data/drug_consumption_prepared.csv).

<div id="pré-processamento"></div>

### Pré-processamento

Neste notebook cada característica do dataset é analisada e são oferecidas explicações para cada escolha tomada. Note que o pré-processamento não inclui a normalização dos dados, apenas a seleção/engenharia de características e transformação dos atributos categóricos.

Esse notebook tem como entrada o arquivo [drug_consumption_prepared](data/drug_consumption_prepared.csv) e gera como saída o arquivo [drug_consumption_preprocessed](data/drug_consumption_preprocessed.csv).

<div id="classificação"></div>

### Classificação

Nesse notebook, observamos várias facetas do problema, tentando encontrar padrões que possam ajudar a obter uma interpretação melhor dos dados.

De início, utilizamos da biblioteca `LazyClassifier` para medir a capacidade de predição de vários classificadores em substâncias relevantes. Dessa forma, podemos filtrar um top $k$ de classificadores e testá-los corretamente na base de dados.

Subsequentemente, são exibidos boxplots e matrizes de confusão que determinam a performance dos classificadores.

<div id="classificação-plus"></div>

### Classificação +

Para continuar investigando possíveis combinações, um notebook extra examina vertentes do problema por aplicar mais configurações aos procedimentos. Os testes incluem:
 
 - Uso de métricas de personalidade somente,
 - Filtro de idades,
 - Redução de dimensionalidade por PCA,
 - Classificadores multiclasse.

Com esses experimentos, procuramos encontrar padrões que possam estar menos reconhecíveis por conta da distribuição dos dados.