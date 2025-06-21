"""
Módulo para engenharia de atributos.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def encode_categorical(df, columns):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[columns])
    return encoded, encoder

def scale_numerical(df, columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    return scaled, scaler
