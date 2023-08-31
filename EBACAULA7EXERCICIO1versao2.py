#!/usr/bin/env python
# coding: utf-8

# # Módulo 07, Tarefa 01
# 
# Vamos começar a mexer na nossa base de projeto? Já fizemos um exercício de montar a variável resposta, já pudemos perceber que essa atividade pode não ser trivial. Vamos agora trabalhar a base para que fique propícia ao *scikitlearn* para trabalharmos.
# 
# Lembrando, a base se chama demo01.csv, e originalmente está publicada [aqui](https://www.kaggle.com/rikdifos/credit-card-approval-prediction).

# #### 1) Carregue a base e avalie:
# 
# - As variáveis
# - Tipos de dados de cada variável
# - Quantidade de missings
# - Distribuição da variável resposta (mau)

# In[15]:


# Carregar a base de dados
data = pd.read_csv('credit_record.csv')  # Substitua pelo caminho correto do arquivo

# Visualizar as primeiras linhas do DataFrame
print(data.head())


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Carregar a base de dados
data = pd.read_csv('credit_record.csv')  # Substitua pelo caminho correto do arquivo

# Variável escolhida para análise
variavel_escolhida = 'STATUS'

# Visualizar as primeiras linhas do DataFrame
print(data.head())

# Verificar os tipos de dados de cada variável
print(data.dtypes)

# Verificar quantidade de valores ausentes (missings) por variável
missing_count = data.isnull().sum()
print(missing_count)

# Plotar a distribuição da variável escolhida
plt.figure(figsize=(8, 6))
data[variavel_escolhida].value_counts().plot(kind='bar')
plt.title(f'Distribuição da Variável {variavel_escolhida}')
plt.xlabel(variavel_escolhida)
plt.ylabel('Contagem')
plt.show()



# #### 2) Vamos montar um metadados
# 
# 1. Crie um dataframe com os nomes de cada variável e o tipo de dados de cada variável.
# 2. Adicione uma coluna nesse *dataframe* chamada "qtd_categorias" e coloque nela o número de categorias correspondente de cada variável. 
#     Dica: 
#         1. inicie uma lista vazia
#         2. faça um for ao longo dos nomes das variáveis, 
#         3. conte o número de categorias dessa variável 
#         4. acumule essa informação de 3. na lista que você criou em 1. 
#         5. No final, essa lista pode ser atribuída à nossa variável.
# 3. Crie variáveis dummy para as variáveis necessárias (i.e. aquelas que são qualitativas e não estão armazenadas como {0, 1} ou {True, False}.

# In[6]:


# Criar um DataFrame de metadados
metadata = pd.DataFrame(data.dtypes, columns=['Tipo de Dados'])

# Adicionar os nomes das variáveis como uma coluna
metadata['Variável'] = metadata.index

# Resetar o índice para que o índice comece de 0
metadata.reset_index(drop=True, inplace=True)

# Mostrar o DataFrame de metadados
print(metadata)


# #### 3) Crie variáveis dummy para as variáveis necessárias (i.e. aquelas que são qualitativas e não estão armazenadas como {0, 1} ou {True, False}. Crie um *dataframe* apenas com as variáveis apropriadas para entrada no scikitlearn - elimine as variáveis tipo *str*, mantendo apenas suas versões *dummy*.

# In[9]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Carregar a base de dados
data = pd.read_csv('credit_record.csv')  # Substitua pelo caminho correto do arquivo

# Selecionar apenas as colunas de tipo 'object' (categóricas)
categorical_columns = data.select_dtypes(include=['object']).columns

# Criar variáveis dummy
data_with_dummies = pd.get_dummies(data, columns=categorical_columns)

# Remover colunas que não são numéricas (tipo 'object')
X = data_with_dummies.select_dtypes(exclude=['object'])

# Agora você pode usar X para treinar modelos no scikit-learn


# #### 4) Qual variável é mais poderosa?
# 
# Considere as variáveis ```possui_email``` e ```posse_de_veiculo```. Faça uma tabela cruzada entre elas e responda qual delas te parece mais poderosa para prever a probabilidade de ```mau = 1```?

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar a base de dados
data = pd.read_csv('credit_record.csv')  # Substitua pelo caminho correto do arquivo

# Substitua 'NOME_DA_VARIAVEL_ALVO' pelo nome da coluna que você deseja usar como variável alvo
target_variable = 'ID'
if target_variable not in data.columns:
    print(f"A coluna '{target_variable}' não foi encontrada no conjunto de dados.")
else:
    # Criar variáveis dummy para todas as colunas categóricas
    data_with_dummies = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns)

    # Definir variáveis de entrada (X) e variável alvo (y)
    X = data_with_dummies.drop(target_variable, axis=1)
    y = data_with_dummies[target_variable]

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar um modelo de regressão logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcular a acurácia do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2f}")


# #### 5) Salve a base, pois ela será utilizada no final deste módulo.

# In[ ]:


# Salvar o DataFrame em um novo arquivo CSV
data.to_csv('novo_arquivo.csv', index=False)  # Substitua 'novo_arquivo.csv' pelo nome desejado


# In[ ]:




