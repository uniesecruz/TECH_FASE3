# preprocessamento.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def substituir_valores(df):
    df = df.replace('?', np.nan)
    df = df.fillna('Unknown')
    return df

class AgrupadorPersonalizado(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['workclass_simplificada'] = X['workclass'].apply(
            lambda x: 'WO_pay' if x in [' Without-pay', ' Never-worked'] else x
        )

        X['occupation_simplificada'] = X['occupation'].apply(
            lambda x: 'Other-service' if x in [' Armed-Forces', 'Other-service'] else x
        )

        def status_married(x):
            if pd.isna(x): return 'Unknown'
            return 'Married' if x in [' Married-civ-spouse', ' Married-AF-spouse'] else (
                'Never-married' if x == ' Never-married' else 'No-spouse'
            )
        X['marital-status_simplificada'] = X['marital-status'].map(status_married)

        def grupos_race(x):
            return 'White' if x == ' White' else 'Other'
        X['race_simplificada'] = X['race'].map(grupos_race)

        def grupos_relacionamento(x):
            return 'Spouse' if x in [' Husband', ' Wife'] else x
        X['relationship_simplificada'] = X['relationship'].map(grupos_relacionamento)

        def grupos_paises(val):
            if pd.isna(val): return 'Unknown'
            if val == ' United-States': return val
            elif val in [' Mexico', ' Cuba', ' Jamaica', ' Puerto-Rico', ' Honduras', ' Columbia', ' Ecuador',
                         ' Guatemala', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Haiti',
                         ' Dominican-Republic', ' El-Salvador', ' Trinadad&Tobago', ' South']:
                return 'America'
            elif val in [' England', ' Canada', ' Germany', ' Italy', ' Poland', ' Portugal', ' France', ' Greece',
                         ' Ireland', ' Hungary', ' Yugoslavia', ' Holand-Netherlands', ' Scotland']:
                return 'Europe'
            elif val in [' India', ' Philippines', ' Iran', ' Cambodia', ' Thailand', ' Laos', ' Taiwan',
                         ' China', ' Japan', ' Vietnam', ' Hong']:
                return 'Asia'
            else:
                return 'Other'
        X['native-country_simplificada'] = X['native-country'].map(grupos_paises)

        return X

def construir_pipeline_preprocessamento():
    variaveis_numericas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    variaveis_binarias = ['sex']  # ❗️ income foi removido
    variaveis_categoricas_simplificadas = [
        'workclass_simplificada', 'occupation_simplificada', 'marital-status_simplificada',
        'relationship_simplificada', 'race_simplificada', 'native-country_simplificada'
    ]

    pipeline = Pipeline(steps=[
        ('substituir', FunctionTransformer(substituir_valores)),
        ('agrupador', AgrupadorPersonalizado()),
        ('transformacoes', ColumnTransformer(transformers=[
            ('binarias', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), variaveis_binarias),
            ('categoricas', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), variaveis_categoricas_simplificadas),
            ('numericas', StandardScaler(), variaveis_numericas)
        ], remainder='drop'))
    ])

    return pipeline
