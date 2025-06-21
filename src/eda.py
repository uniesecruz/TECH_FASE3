"""
Módulo para análise exploratória de dados (EDA).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, bins=30):
    ax = df[column].hist(bins=bins)
    ax.set_title(f'Distribuição dos dados da variável {column}')
    ax.set_ylabel('Número de pessoas')
    ax.set_xlabel(column)
    plt.show()

def plot_boxplot(df, y, x=None):
    if x:
        sns.boxplot(y=y, x=x, data=df)
    else:
        sns.boxplot(y=df[y])
    plt.show()
