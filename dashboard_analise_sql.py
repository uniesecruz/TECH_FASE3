import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sqlalchemy import create_engine

# Função para decodificar o rótulo binário
def decodificar_income(y_binario):
    return np.where(y_binario == 1, '>50K', '<=50K')

# Carrega modelo e pipeline
@st.cache_resource
def carregar_modelo_pipeline():
    with open("modelo_lr.pkl", "rb") as f_modelo:
        modelo = pickle.load(f_modelo)
    with open("pipeline_preprocessamento.pkl", "rb") as f_pipeline:
        pipeline = pickle.load(f_pipeline)
    return modelo, pipeline

modelo, pipeline = carregar_modelo_pipeline()

# Dados da conexão com SQL Server
server = 'DESKTOP-H24E0EB'
database = 'MLE'
driver = 'ODBC Driver 17 for SQL Server'
connection_string = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&Trusted_Connection=yes&TrustServerCertificate=yes"
engine = create_engine(connection_string)

st.title("📊 Análise de Classificação com Dados do Banco")

if st.button("🔄 Carregar Dados do Banco"):
    try:
        df = pd.read_sql("SELECT * FROM dbo.MLE_teste", engine)
        st.subheader("🔍 Pré-visualização dos dados")
        st.write(df.head())

        if 'income' not in df.columns:
            st.error("A coluna 'income' não foi encontrada na tabela.")
        else:
            y_true_labels = df["income"]
            X_raw = df.drop(columns=["income"], errors="ignore")

            # Aplica pipeline
            X_processado = pipeline.transform(X_raw)
            y_pred_binario = modelo.predict(X_processado)

            # Decodifica valores binários para rótulos
            y_pred = decodificar_income(y_pred_binario)
            y_true = y_true_labels.values

            # Métricas
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label='>50K')
            rec = recall_score(y_true, y_pred, pos_label='>50K')
            f1 = f1_score(y_true, y_pred, pos_label='>50K')

            st.subheader("📈 Métricas de Classificação")
            st.write(f"**Acurácia:** {acc:.4f}")
            st.write(f"**Precisão:** {prec:.4f}")
            st.write(f"**Recall:** {rec:.4f}")
            st.write(f"**F1-Score:** {f1:.4f}")

            # Matriz de confusão
            st.subheader("🔀 Matriz de Confusão")
            labels = ['<=50K', '>50K']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_title("Matriz de Confusão")
            st.pyplot(fig)

            # Distribuição das classes preditas
            st.subheader("📊 Distribuição das Previsões")
            pred_df = pd.DataFrame({'Real': y_true, 'Predito': y_pred})
            fig2, ax2 = plt.subplots()
            pred_df['Predito'].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title("Distribuição das classes preditas")
            ax2.set_ylabel("Frequência")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao acessar o banco de dados: {e}")
