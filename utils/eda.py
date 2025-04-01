import pandas as pd
def Map_Var_DF (features, df):
  #Criando um dicionário para receber as variáveis
  dict_var = {"feature": [],
              "Tipo": [],
              "Categórico": [],
              "Binário": [],
              "Qtd var unico": [],
              "Min": [],
              "Max": [],
              "% Qtd de Nulos": []}

  #Criando um loop a partir das features
  for feature in features:

    #Armazenando o nome da feature
    dict_var['feature'].append(feature)

    #Armazenando o tipo da variável
    dict_var['Tipo'].append(df[feature].dtypes)

    #Armazenando a quantidade de valores nulos
    dict_var['% Qtd de Nulos'].append(round(df[feature].isnull().sum() / df.shape[0],4))

    if ((df[feature].dtype == "O")):

      #Atribuindo o valor 1 se a variável for categórica
      dict_var['Categórico'].append(1)

      #Armazenando a quantidade de valores únicos
      dict_var['Qtd var unico'].append(df[feature].nunique())

      #Armazenando os valores mínimos
      dict_var['Min'].append("N/A")

      #Armazenando os valores máximos
      dict_var['Max'].append("N/A")

      if (df[feature].nunique() == 2):

        #Atribuindo o valor 1 se a variável for binária
        dict_var['Binário'].append(1)

      else:

        #Atribuindo o valor 0 se a variável não for binária
        dict_var['Binário'].append(0)

    else:

      #Atribuindo o valor 0 se a variável não for categórica
      dict_var['Categórico'].append(0)

      #Armazenando a quantidade de valores únicos
      dict_var['Qtd var unico'].append(df[feature].nunique())

      #Atribuindo o valor 0 se a variável não for binária
      dict_var['Binário'].append(0)

      #Armazenando os valores mínimos
      dict_var['Min'].append(df[feature].min())

      #Armazenando os valores máximos
      dict_var['Max'].append(df[feature].max())

  #Transformando o dicionário em dataframe
  df_var = pd.DataFrame.from_dict(data = dict_var)

  #Imprimindo o dataframe
  return df_var

def descricao_dataset(df):
  ##descrição do dataset
  print('Quantidade de dados:  %s\n' % (df.shape, ))
  print('Quantidade de variáveis: %s\n' % df.columns)

  print('Dataframe primeira ate a ultima linha:\n' % df.columns)
  display(df)

  print('\nDescrição das Colunas:\n')
  display(df.info())
  display(df.describe())

  print('\nValores Nulos:\n')
  display(df.isnull().sum())
  
# Função criada para auxiliar no processo de analise das variaveis categoricas. Obtendo os valores totais, porcentagem e valores nulos.

def analise_variavel_categorica(df, nome_variavel):
  ##nome_variavel é o nome da variavel analisada
  print('valores dos dados pela classe de variavel  %s:\n%s\n' % (nome_variavel, df[nome_variavel].value_counts(dropna=False))) #Valores totais da variavel por categoria
  print('Percentagem dos dados pela classe de variavel %s:\n%s\n' % (nome_variavel, df[nome_variavel].value_counts(normalize=True, dropna=False))) #porcentagem da variavel por categoria
  print('Valores unicos da variavel %s:\n%s\n'  % (nome_variavel, df[nome_variavel].unique())) #valores unicos da variavel

# Função criada para auxiliar no processo de analise das variaveis continuas. Obtendo os valores totais, porcentagem e valores nulos.
def analise_variavel_continua(df, nome_variavel):
  ##nome_variavel é o nome da variavel analisada
  print('Distribuição de dados da variavel  %s:\n%s\n' % (nome_variavel, df[nome_variavel].value_counts(dropna=False))) #distribuição dos valores da variavel
  print('Porcentagem da distribuição dos dados da variavel  %s:\n%s\n' % (nome_variavel, df[nome_variavel].value_counts(normalize=True, dropna=False))) #porcentagem de distribuição dos valores da variavel
  print('Total de valores unicos da variavel %s: %i\n'  % (nome_variavel, len(df[nome_variavel].unique()))) # valores unicos da variavel
  display(df[nome_variavel].describe())