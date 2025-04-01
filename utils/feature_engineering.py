from sklearn.preprocessing import StandardScaler
def padronizacao_variavel(df_x,variavel):
    ## Padronizando as variaveis
    std_sca=StandardScaler()
    std_sca.fit(df_x[variavel])
    df_x.loc[:,variavel]=std_sca.transform(df_x[variavel])
    return df_x