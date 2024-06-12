# Drug Consumption Classifier

Com este projeto, nós procuramos encontrar relações entre usuários de substâncias e algumas características pessoais e comportamentais.

Este repositório contém o pré-processamento, as análises e alguns testes de classificadores no dataset.

## Dataset

O dataset utilizado neste projeto pode ser encontrado [aqui](https://www.kaggle.com/datasets/mexwell/drug-consumption-classification). O [notebook de pré-processamento](https://www.kaggle.com/code/mexwell/starter-notebook-convert-column-values) que foi utilizado como base para processar os dados originais também foi disponibilizado pelo autor original.

# Executando os Experimentos

Para executar os notebooks, é necessário instalar alguns pacotes:

```bash
pip install requirements.txt
```

## Pré-processamento

Na pasta `data/`, existem duas versões do dataset: a original e a pré-processada (note que o pré-processamento não inclui a normalização dos dados).

Dessa forma, o uso do notebook de pré-processamento não é necessário, mas serve para entender a nova organização do dataset.

Cada característica do dataset é analisada e são oferecidas explicações para cada escolha feita sobre os valores apresentados no dataset utilizado nos experimentos.

## Classificação

Nesse notebook, observamos várias facetas do problema, tentando encontrar padrões que possam ajudar a obter uma interpretação melhor dos dados.

De início, utilizamos da biblioteca `LazyClassifier` para medir a capacidade de predição de vários classificadores em substâncias relevantes. Dessa forma, podemos filtrar um top $k$ de classificadores e testá-los corretamente na base de dados.

Subsequentemente, são exibidos boxplots e matrizes de confusão que determinam a performance dos classificadores.

## Classificação +

Para continuar investigando possíveis combinações, um notebook extra examina vertentes do problema por aplicar mais configurações aos procedimentos. Os testes incluem:
 
 - Filtro de características,
 - Seleção de samples,
 - Aplicação de redução de dimensionalidade,
 - Classificadores multiclasse.

Com esses experimentos, procuramos encontrar padrões que possam estar menos reconhecíveis por conta da distribuição dos dados.