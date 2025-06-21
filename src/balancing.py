"""
Módulo para balanceamento de classes.
"""
#
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
