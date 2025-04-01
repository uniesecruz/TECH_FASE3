
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#Função criada para auxiliar avaliação do desempenho dos modelos

def calc_performance_metrics(nome_modelo, validacao_y, preditora):
  ##metricas de classificação
  calc_acuracia=accuracy_score(validacao_y, preditora)
  calc_revocacao=recall_score(validacao_y, preditora)
  calc_precisao=precision_score(validacao_y, preditora)
  calc_f1=f1_score(validacao_y, preditora)

  print(nome_modelo+' Metrica de classificação:')
  print('Acuracia:', calc_acuracia)
  print('Revocacao:', calc_revocacao)
  print('Precisao:', calc_precisao)
  print('F1:', calc_f1)
  return calc_acuracia, calc_revocacao, calc_precisao, calc_f1

def calc_confusion_matrix(validacao_y, preditora):
  ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(validacao_y, preditora), display_labels=['<=$50K', '>$50K']).plot()
  tn, fp, fn, tp=confusion_matrix(validacao_y, preditora).ravel()
  print('\nTN: ', tn, 'FP: ', fp, 'FN: ', fn, 'TP: ', tp)