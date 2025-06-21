"""
Funções utilitárias para métricas e visualizações.
"""
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calc_performance_metrics(nome_modelo, y_true, y_pred):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    print(f'\n{nome_modelo} - Métricas de Classificação:')
    print('Acurácia:', round(accuracy_score(y_true, y_pred), 4))
    print('Revocação:', round(recall_score(y_true, y_pred), 4))
    print('Precisão:', round(precision_score(y_true, y_pred), 4))
    print('F1 Score:', round(f1_score(y_true, y_pred), 4))

def plot_confusion_matrix(nome_modelo, y_true, y_pred, labels=['<=50K', '>50K']):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.show()
