import streamlit as st
import pandas as pd
import pickle

# Função de pré-processamento (mesma usada no treino)
def preprocessar_dados(df):
    df.replace('?', pd.NA, inplace=True)
    df.fillna('Unknown', inplace=True)

    df['workclass_simplificada'] = df['workclass'].apply(lambda v: 'WO_pay' if v in [' Without-pay', ' Never-worked'] else v)
    df['occupation_simplificada'] = df['occupation'].apply(lambda v: 'Other-service' if v in [' Armed-Forces', 'Other-service'] else v)

    def status_married(v):
        if pd.isna(v): return 'Unknown'
        return 'Married' if v in [' Married-civ-spouse', ' Married-AF-spouse'] else ('Never-married' if v == ' Never-married' else 'No-spouse')
    df['marital-status_simplificada'] = df['marital-status'].map(status_married)

    def grupos_race(v):
        if pd.isna(v): return 'Unknown'
        return 'White' if v == ' White' else 'Other'
    df['race_simplificada'] = df['race'].map(grupos_race)

    def grupos_relacionamento(v):
        if pd.isna(v): return 'Unknown'
        return 'Spouse' if v in [' Husband', ' Wife'] else v
    df['relationship_simplificada'] = df['relationship'].map(grupos_relacionamento)

    def grupos_paises(val):
        if pd.isna(val): return 'Unknown'
        if val == ' United-States': return val
        elif val in [' Mexico', ' Cuba', ' Jamaica', ' Puerto-Rico', ' Honduras', ' Columbia', ' Ecuador', ' Guatemala', ' Peru', 
                     ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Haiti', ' Dominican-Republic', ' El-Salvador', ' Trinadad&Tobago', ' South']:
            return 'America'
        elif val in [' England', ' Canada', ' Germany', ' Italy', ' Poland', ' Portugal', ' France', ' Greece', ' Ireland', ' Hungary', ' Yugoslavia', ' Holand-Netherlands', ' Scotland']:
            return 'Europe'
        elif val in [' India', ' Philippines', ' Iran', ' Cambodia', ' Thailand', ' Laos', ' Taiwan', ' China', ' Japan', ' Vietnam', ' Hong']:
            return 'Asia'
        else:
            return 'Other'

    df['native-country_simplificada'] = df['native-country'].map(grupos_paises)

    cols_cat = [
        'workclass_simplificada', 'occupation_simplificada', 'marital-status_simplificada',
        'relationship_simplificada', 'race_simplificada', 'native-country_simplificada'
    ]

    df = pd.get_dummies(df, columns=cols_cat, drop_first=True)

    cols_drop = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    df.drop(columns=[col for col in cols_drop if col in df.columns], inplace=True)

    return df

# Carregar o modelo treinado
with open("modelo_lr.pkl", "rb") as f:
    modelo = pickle.load(f)

# Streamlit App
st.title("📊 Classificador de Renda (>50K)")

st.markdown("Insira os dados abaixo para prever se a renda de uma pessoa é maior que 50K:")

# Entradas do usuário
age = st.number_input("Idade", min_value=17, max_value=90, value=82)
fnlwgt = st.number_input("Peso Final (fnlwgt)", value=132870)
education_num = st.number_input("Educação (numérica)", min_value=1, max_value=20, value=9)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=4356)
hours_per_week = st.number_input("Horas por semana", min_value=1, max_value=100, value=18)

workclass = st.selectbox("Classe de trabalho", [' Private', ' Self-emp-not-inc', ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked'])
education = st.selectbox("Educação", [' Bachelors', ' HS-grad', ' Some-college', ' Masters'])
marital_status = st.selectbox("Estado civil", [' Never-married', ' Married-civ-spouse', ' Divorced', ' Separated', ' Widowed'])
occupation = st.selectbox("Ocupação", [' Exec-managerial', ' Craft-repair', ' Sales', ' Adm-clerical', ' Armed-Forces', 'Other-service'])
relationship = st.selectbox("Relacionamento", [' Not-in-family', ' Husband', ' Own-child', ' Wife'])
race = st.selectbox("Raça", [' White', ' Black', ' Asian-Pac-Islander', ' Other'])
sex = st.selectbox("Sexo", [' Male', ' Female'])
native_country = st.selectbox("País de origem", [' United-States', ' Mexico', ' Philippines', ' India', ' Germany', ' Other'])

# Criar DataFrame de entrada
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

# Aplicar engenharia de atributos
entrada_processada = preprocessar_dados(entrada_df)

# Corrigir colunas que não existem
for col in modelo.feature_names_in_:
    if col not in entrada_processada.columns:
        entrada_processada[col] = 0

# Garantir ordem correta das colunas
entrada_processada = entrada_processada[modelo.feature_names_in_]

# Previsão
if st.button("Realizar Predição"):
    pred = modelo.predict(entrada_processada)[0]
    resultado = ">50K" if pred == 1 else "<=50K"
    st.success(f"✅ Renda prevista: {resultado}")
