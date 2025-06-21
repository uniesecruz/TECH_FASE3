"""
Módulo para download e carregamento dos dados.
"""

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_ref, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=data_dir, unzip=True)

def load_csv(file_path, columns=None):
    return pd.read_csv(file_path, names=columns, header=None, index_col=False, skiprows=1)

