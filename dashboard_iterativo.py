import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# ---------------------- Streamlit ----------------------

st.set_page_config(page_title="Classificador de Renda", layout="centered")
st.title("📊 Classificador de Renda (>50K)")
st.markdown("Preencha os dados abaixo para prever se a renda de uma pessoa é maior que 50K:")

# Inputs
age = st.number_input("Idade", min_value=17, max_value=90, value=35)
fnlwgt = st.number_input("Peso Final (fnlwgt)", value=140000)
education_num = st.number_input("Educação (numérica)", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.number_input("Horas por semana", min_value=1, max_value=100, value=40)

workclass = st.selectbox("Classe de trabalho", [' Private', ' Self-emp-not-inc', ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked'])
education = st.selectbox("Educação", [' Bachelors', ' HS-grad', ' Some-college', ' Masters'])
marital_status = st.selectbox("Estado civil", [' Never-married', ' Married-civ-spouse', ' Divorced', ' Separated', ' Widowed'])
occupation = st.selectbox("Ocupação", [' Exec-managerial', ' Craft-repair', ' Sales', ' Adm-clerical', ' Armed-Forces', 'Other-service'])
relationship = st.selectbox("Relacionamento", [' Not-in-family', ' Husband', ' Own-child', ' Wife'])
race = st.selectbox("Raça", [' White', ' Black', ' Asian-Pac-Islander', ' Other'])
sex = st.selectbox("Sexo", [' Male', ' Female'])
native_country = st.selectbox("País de origem", [' United-States', ' Mexico', ' Philippines', ' India', ' Germany', ' Other'])

# Criar DataFrame
entrada_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'education-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}])

st.subheader("📌 Dados de Entrada")
st.dataframe(entrada_df)

# Carrega modelo e pipeline
with open("modelo_lr.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("pipeline_preprocessamento.pkl", "rb") as f:
    pipeline = pickle.load(f)

entrada_df = entrada_df.drop(columns=['income'], errors='ignore')
entrada_processada = pipeline.transform(entrada_df)


# Botão de predição
if st.button("🚀 Realizar Predição"):
    pred = modelo.predict(entrada_processada)[0]
    prob = modelo.predict_proba(entrada_processada)[0]

    resultado = ">50K" if pred == 1 else "<=50K"
    st.success(f"✅ Renda prevista: {resultado}")

    st.subheader("📈 Probabilidade da Predição")
    prob_df = pd.DataFrame({
        "income": ["<=50K", ">50K"],
        "Probabilidade": prob
    })

    fig, ax = plt.subplots()
    ax.bar(prob_df["income"], prob_df["Probabilidade"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidade")
    st.pyplot(fig)

    if hasattr(modelo, "coef_"):
        st.subheader("🧠 Impacto das Variáveis no Modelo")
        try:
            nomes_features = modelo.feature_names_in_
        except AttributeError:
            nomes_features = [f"Var{i}" for i in range(len(modelo.coef_[0]))]

        importances = pd.Series(modelo.coef_[0], index=nomes_features)
        importances = importances.sort_values(key=abs, ascending=False).head(15)

        fig2, ax2 = plt.subplots()
        importances.plot(kind="barh", ax=ax2)
        ax2.set_title("Top 15 Variáveis Mais Influentes")
        ax2.invert_yaxis()
        st.pyplot(fig2)
