# imports
import streamlit as st
import pandas as pd
import pickle as pkl
import os

# Import bibliotecas básicas
import numpy as np
from numpy import mean
from scipy import stats
import pandas as pd
from datetime import  timedelta

# Import bibliotecas de visualização e manipulação
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Import XGBoost
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="pastel")

PATH = os.getcwd()

# Lista global para armazenar os resultados
index_test = None
evaluation_results = {
    'MAE': {},
    'MSE': {},
    'MAPE': {},
    'R2': {}
}

# Helper Functions
def correct_value_with_separators(row):
    value = row['Preco']
    before_dot = str(value)[:-2]
    after_dot = str(value)[-2:]
    new_value = before_dot + "." + after_dot
    return new_value

def correct_value_with_separators(row):
    value = row['Preco']
    before_dot = str(value)[:-2]
    after_dot = str(value)[-2:]
    new_value = before_dot + "." + after_dot
    return new_value

def calculate_moving_averages(df, target, windows): #adicionar colunas de média móvel ao dataframe para modelo
    for window in windows:
        ma_col_name = f'{target}_ma_{window}'
        df[ma_col_name] = df[target].rolling(window=window).mean() #.dropna()
        
    df.dropna(inplace=True)
    return df

def calculate_moving_averages_all(df, target, windows): #adicionar colunas de média móvel ao dataframe para modelo
# função de calculo para previsões futuras
    for window in windows:
        ma_col_name = f'{target}_ma_{window}'
        df[ma_col_name] = df[target].rolling(window=window).mean() #.dropna()

    # inserindo a média dos dados móveis dos ultimos dias com valore conhecido no modelo
    for ma in [f'{target}_ma_{window}' for window in windows]:
        df[ma].fillna(method='ffill', inplace=True)

    return df

def agg_datasets_train_test(df, y_train, y_test, prediction):
    global index_test 
    index_train = df.index[:len(y_train)]
    index_test = df.index[len(y_train):len(y_train) + len(y_test)]
    
    # Criando DataFrames para os conjuntos de treino, teste e previsão
    df_train = pd.DataFrame({'y_train': y_train}, index=index_train)
    df_test = pd.DataFrame({'y_test': y_test}, index=index_test)
    df_predict = pd.DataFrame({'predict': prediction}, index=index_test)
    return df_train, df_test, df_predict

def evaluate_model(y_test, prediction, modelo):
  mae = mean_absolute_error(y_test, prediction)
  mse = mean_squared_error(y_test, prediction)
  mape = mean_absolute_percentage_error(y_test, prediction)
  r2 = r2_score(y_test, prediction)
  
  # Adiciona os resultados ao dicionário de métricas
  evaluation_results['MAE'][modelo] = mae
  evaluation_results['MSE'][modelo] = mse
  evaluation_results['MAPE'][modelo] = mape
  evaluation_results['R2'][modelo] = r2

  print(f"Métricas de avaliação {modelo}:")
  print(f"\tMAE: {mean_absolute_error(y_test, prediction)}")
  print(f"\tMSE: {mean_squared_error(y_test, prediction)}")
  print(f"\tMAPE: {mean_absolute_percentage_error(y_test, prediction):.2f}%")
  print(f"\tR2 Score: {r2_score(y_test, prediction):.2f}%")
  
  mae = mean_absolute_error(y_test, prediction)
  mse = mean_squared_error(y_test, prediction)
  mape = mean_absolute_percentage_error(y_test, prediction)
  r2 = r2_score(y_test, prediction)
  return mae, mse, mape, r2
 
def evaluate_model_grid(y_test, prediction, modelo):
  mae = mean_absolute_error(y_test, prediction)
  mse = mean_squared_error(y_test, prediction)
  mape = mean_absolute_percentage_error(y_test, prediction)
  r2 = r2_score(y_test, prediction)
  
  # Adiciona os resultados ao dicionário de métricas
  evaluation_results['MAE'][modelo] = mae
  evaluation_results['MSE'][modelo] = mse
  evaluation_results['MAPE'][modelo] = mape
  evaluation_results['R2'][modelo] = r2
  
  print(f"Métricas de avaliação {modelo}:")
  print(f"\tMAE: {mean_absolute_error(y_test, prediction)}")
  print(f"\tMSE: {mean_squared_error(y_test, prediction)}")
  print(f"\tMAPE: {mean_absolute_percentage_error(y_test, prediction):.2f}%")
  print(f"\tR2 Score: {r2_score(y_test, prediction):.2f}%")
  print("Melhores parâmetros:")
  print(f"\tMelhores hiperparâmetros encontrados: {grid_search.best_params_}")
  print(f"\tMelhor pontuação: {grid_search.best_score_:.2f}%")

def plot_predictions_grid(x_test, y_test, prediction, modelo):
# plot predição geral
  df_test = pd.DataFrame({"date": x_test, "actual": y_test, "prediction": prediction })
  figure, ax = plt.subplots(figsize=(10, 5))
  df_test.plot(ax=ax, label="Actual", x="date", y="actual")
  df_test.plot(ax=ax, label="Prediction", x="date", y="prediction")
  plt.title(f'Performance dados reais x preditos {modelo}')
  plt.legend(["Actual", "Prediction"])
  plt.show()

def plot_prediction_test(df_train, df_test, df_predict, modelo):
# gráfico plot dados teste 
  plt.figure(figsize=(15, 6))
  plt.plot(df_train['y_train'], label='Treino')
  plt.plot(df_test['y_test'], label='Teste', linewidth=2, color = "gray")  # Linha mais grossa
  plt.plot(df_predict['predict'], label='Previsão', linewidth=0.3,color='red', linestyle='--')  # Linha pontilhada
  plt.title(f'Comparação entre Treino, Teste e Previsão {modelo}')
  plt.xlabel('Data')
  plt.ylabel('Preço')
  plt.legend()
  plt.show()
  
def plot_residuos(y_test, prediction, modelo):
# gráfico avalicação de resíduos
  residuos = y_test - prediction
  df_residuos = pd.DataFrame({'residuos': residuos}, index=index_test)  # Criando DataFrame para os resíduos
  plt.figure(figsize=(15, 6))
  plt.plot(df_residuos['residuos'], label='Resíduos', linestyle='-', color='blue')
  plt.title(f'Análise dos Resíduos - {modelo}')
  plt.xlabel('Data')
  plt.ylabel('Resíduos')
  plt.axhline(y=0, color='r', linestyle='--')  # Linha horizontal em y=0
  plt.legend()
  plt.show()

def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week  
    return df

def time_series_xgboost_ipeadata(dataframe: pd.DataFrame, periodo: str):
    dados_modelo = ipeadata.sort_values(by=["Data"],ascending=True) # ordenando as datas em forma crescente
    dados_modelo.columns = ['data', 'preco']
    dados_xgb_datas = dados_modelo.copy().set_index('data') # criação de uma cópia do dataframe para manipulação, deixando coluna "data como index
    dados_xgb_datas = dados_xgb_datas.sort_values(by=["data"],ascending=True) # ordenando as datas em 
    cols = list(dados_xgb_datas.columns.drop("preco"))

    # criando features de datas
    dados_xgb_datas = create_features(dados_xgb_datas)
    windows = [3, 7, 14, 21] #janelas para médias móveis em dias
    dados_xgb_datas = calculate_moving_averages(dados_xgb_datas, 'preco', windows)
    
    dados_xgb_datas = dados_xgb_datas.sort_values(by=["data"], ascending=True)

    x = dados_xgb_datas.drop('preco', axis=1)
    y = dados_xgb_datas['preco']

    model = xgb.XGBRegressor(n_estimators=700,
                            eta = 0.01, # = learning_rate
                            max_depth = 5,
                            objective='reg:linear',
                            colsample_bytree = 0.7)

    # model.fit(x_train, y_train, verbose = False)
    model.fit(x, y, 
            eval_set=[(x, y)],
          verbose = 100)
    
    dados_xgb_valid = dados_xgb_datas.copy()
    features = dados_xgb_datas.columns[1:]

    period = 0
    if periodo == 'semanal':
        period += 7
    elif periodo == 'quinzenal':
        period += 15
    elif periodo == 'mensal':
        period += 30
        
    future_days = pd.date_range(start=dados_xgb_valid.index[-1] + pd.Timedelta(days=1), periods=period, freq="1d")
    future_df = pd.DataFrame(index = future_days)
    future_df['isFuture'] = True
    dados_xgb_valid['isFuture'] = False
    df_and_future = pd.concat([dados_xgb_valid, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = calculate_moving_averages_all(df_and_future, 'preco', windows)
    
    data_corte = dados_xgb_valid.index.max() - timedelta(days=60)
    df_and_future = df_and_future[df_and_future.index >= data_corte]
    df_and_future['pred'] = model.predict(df_and_future[features])
    
    return df_and_future
    

try:
    ipeadata = pd.read_html(r"http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view", encoding='utf-8', header=0)[2]
    ipeadata.to_pickle(os.path.join(PATH, 'ipeadata.pkl'))
    
except Exception:
    ipeadata = pd.read_pickle(os.path.join(PATH, 'ipeadata.pkl'))

# Tratamento no Dataframe
ipeadata.columns = ['Data', 'Preco']

ipeadata['Data'] = pd.to_datetime(ipeadata['Data'], format='%d/%m/%Y')
ipeadata.Preco = ipeadata.apply(lambda row: correct_value_with_separators(row), axis=1)
ipeadata.Preco = ipeadata.Preco.astype(float)


st.header('Modelo Time Series')


periodo = st.selectbox('Selecione o Período que deseja prever os valores do Ipeadata: ', ['Selecione um valor....','Semanal', 'Quinzenal', 'Mensal'])

if periodo != 'Selecione um valor....':
    if periodo != 'Semanal':
        st.warning('Para os períodos mais distantes de semanal, a tendência é que a acurácia diminua.',icon='⚠️')
    with st.spinner('Gerando previsão...'):
        df_and_future = time_series_xgboost_ipeadata(dataframe=ipeadata, periodo=str(periodo).lower())
    st.success('Previsão finalizada!', icon="✅")

    df_valid = df_and_future.loc[~df_and_future.preco.isna()]
    metrics = evaluate_model(df_valid['preco'], df_valid.pred.round(2), 'Previsão')
    
    st.subheader('Métricas de Avaliação das Previsões')
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric('MAE: ', round(metrics[0], 2))
    col2.metric('MS2: ', round(metrics[1], 2))
    col3.metric('MAPE: ', round(metrics[2], 2))
    col4.metric('R2 Score: ', str(round(metrics[3] * 100, 2))+"%")
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_and_future.index, y=df_and_future['preco'], mode='lines', name='valores reais'))
    fig.add_trace(go.Scatter(x=df_and_future.index, y=df_and_future['pred'], mode='lines', name='valores preditos'))
    
    st.plotly_chart(fig)
else:
    pass
