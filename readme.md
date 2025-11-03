# Projeto Final da Disciplina de Introdução a Ciência de dados

Projeto de Introdução à Ciência de Dados — análise e modelos preditivos sobre o dataset California Housing.

**Autora**: Amanda Rodrigues de Souza  
**Material base**: Professor José Eduardo Santarem Segundo
**Data**: 03/11/2025

## Conteúdo
- Notebook principal: [Projeto_Int_DS.ipynb](Projeto_Int_DS.ipynb)  
- Dados: [Dados/california_housing_train.csv](Dados/california_housing_train.csv) 

## Descrição rápida
O notebook [Projeto_Int_DS.ipynb](Projeto_Int_DS.ipynb) realiza:
- Importação e inspeção dos dados.
- Análise exploratória (boxplot, mapa de calor).
- Separação de variáveis.
- Treino e avaliação de modelos:
    - KNN;
    - Decision Tree;
    - Random Forest.


## Como executar
1. Abrir o notebook [Projeto_Int_DS.ipynb](Projeto_Int_DS.ipynb) no Jupyter / VS Code.
2. Certificar-se de que o arquivo de dados [Dados/california_housing_train.csv](Dados/california_housing_train.csv) está presente no diretório `Dados/`.
3. Executar as células em ordem. Requisitos Python (instalar via pip/conda):
   - pandas, numpy, seaborn, matplotlib, scikit-learn

## Principais símbolos no notebook
- [`dados`](Projeto_Int_DS.ipynb) — dataframe com os dados carregados.  
- [`X`](Projeto_Int_DS.ipynb) — dataframe com as features selecionadas.  
- [`y`](Projeto_Int_DS.ipynb) — série com a variável alvo (median_house_value).  
- [`features`](Projeto_Int_DS.ipynb) — lista de colunas usadas como entrada.  
- [`modelo`](Projeto_Int_DS.ipynb), [`modelo2`](Projeto_Int_DS.ipynb) — modelos KNN.  
- [`modelotree`](Projeto_Int_DS.ipynb) — Decision Tree.  
- [`modelorf`](Projeto_Int_DS.ipynb) — Random Forest.  

## Observações
- O notebook contém comentários em português e células de visualização dos resultados.  
- Para reproduzir exatamente os resultados, usar o mesmo random_state presente nas células de treino.