import streamlit as st
import datetime
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle as pkl
import os

# Import bibliotecas básicas
import numpy as np
from numpy import mean
from scipy import stats
from datetime import  timedelta

# Import bibliotecas de visualização e manipulação
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
    

def apply_model(df_fob:pd.DataFrame):
    #df_fob['data'] = pd.to_datetime(df_fob['data'], format='%d/%m/%Y')
    #df_fob['data'] = df_fob['data'].dt.date
    df_fob['ano'] = df_fob['data'].dt.year
    df_fob['aumento'] = df_fob['preco'].diff()
    df_fob['aceleracao'] = df_fob['aumento'].diff()
    dia_semana = {
        0: 'Segunda',
        1:'Terça',
        2:'Quarta',
        3:'Quinta',
        4:'Sexta',
        5:'Sabado',
        6:'Domingo'
    }
    df_fob['dia_semana'] = df_fob['data'].dt.weekday
    df_fob['dia_semana'] = df_fob['dia_semana'].map(dia_semana)
    return df_fob

def format_date(date:datetime.datetime):
    """
    Recebe uma valor do tipo e data e retorna 
    a data no formato DD/MM/AAAA
    """
    str_data = str(date.day) + '/' + str(date.month) + '/' + str(date.year)
    return str_data

def conv_float(s):
    try:
        # Verifica se a string tem pelo menos duas casas decimais
        if len(s) < 2:
            raise ValueError("String não contém duas casas decimais.")

        # Extrai as últimas duas casas decimais
        decimal_part = s[-2:]

        # Substitui o ponto pela parte decimal e converte para float
        resultado = float(s[:-2] + '.' + decimal_part)

        return resultado
    except ValueError as e:
        print(f"Erro: {e}")
        return None

def transform_dataframe_web():
    try:
        dfs_fob = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view')
        df_fob = dfs_fob[2]
        df_fob.columns = ['data', 'preco']
        df_fob.drop(index=0, inplace=True)
        df_fob['preco'] = df_fob['preco'].apply(conv_float)
        df_fob['data'] = pd.to_datetime(df_fob['data'], format='%d/%m/%Y')
        return df_fob
    except:
        file = open('./df_fob.p', 'rb')
        df_fob = pickle.load(file)
        file.close()
        st.warning('Não foi possível atualizar os dados')
        df_fob = apply_model(df_fob)
        return df_fob

def save_pickle(df:pd.DataFrame):
    caminho_do_arquivo = 'df_fob.p'
    delete_pickle()
    try:
        file = open(caminho_do_arquivo, 'wb')
        pickle.dump(df_fob, file)
        file.close()
    except:
        print('Não foi possível salvar o arquivo')
        pass

def delete_pickle():
    if os.path.exists(caminho_do_arquivo):
        os.remove(caminho_do_arquivo)
        print(f"Arquivo '{caminho_do_arquivo}' deletado com sucesso.")
    else:
        print(f"O arquivo '{caminho_do_arquivo}' não existe.")

def criar_figura_matplotlib(dataframe, x_col, y_col, tipo_grafico='scatter', titulo=None, xlabel=None, ylabel=None):
    """
    Cria um objeto Figure da biblioteca Matplotlib a partir de um DataFrame.

    Parâmetros:
    - dataframe: DataFrame pandas contendo os dados.
    - x_col: Nome da coluna para o eixo x.
    - y_col: Nome da coluna para o eixo y.
    - tipo_grafico: Tipo de gráfico (por padrão, 'scatter').
    - titulo: Título do gráfico (opcional).
    - xlabel: Rótulo do eixo x (opcional).
    - ylabel: Rótulo do eixo y (opcional).

    Retorna:
    - Objeto Figure da biblioteca Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    eventos = {
        'Crise Financeira de 2008': '2008-01-01',
        'Primavera Árabe': '2011-01-01',
        'Pandemia de COVID-19': '2020-01-01',
        'Tendência de Energias Renováveis': '2020-01-01'  # Supondo que a tendência começou em 2020
    }
    # Adicione condições para outros tipos de gráficos, se necessário
    if tipo_grafico == 'hist':
        ax.hist(dataframe[y_col], bins=20)
    elif tipo_grafico == 'bar':
        ax.bar(dataframe[x_col], dataframe[y_col])
    elif tipo_grafico == 'line':
        ax.plot(dataframe[x_col], dataframe[y_col])
        # for evento, data in eventos.items():
        #     ax.axvline(pd.to_datetime(data), linestyle='--', label=evento)
        #     ax.legend()
    # Adicione mais opções conforme necessário
    ax.grid()
    ax.set_title(titulo, {'color': 'red'})
    ax.set_xlabel(xlabel, {'color': 'red'})
    ax.set_ylabel(ylabel, {'color': 'red'})

    return fig

dic_group_keys = {
                'Anual': 'ano',
                'Mensal': 'mes',
                'Semanal': 'WeekNumber'
            }

def group_sum(df, periodo):
    df = df[['preco', dic_group_keys[periodo]]]
    df_fob_grouped = df.groupby(dic_group_keys[periodo]).sum()
    df_fob_grouped['x'] = df_fob_grouped.index 
    return df_fob_grouped

def group_mean(df, periodo):
    df = df[['preco', dic_group_keys[periodo]]]
    df_fob_grouped = df.groupby(dic_group_keys[periodo]).mean()
    df_fob_grouped['x'] = df_fob_grouped.index 
    return df_fob_grouped

def group_min(df, periodo):
    df_fob_grouped = df.groupby(dic_group_keys[periodo]).min()
    df_fob_grouped['x'] = df_fob_grouped.index 
    return df_fob_grouped

def group_max(df, periodo):
    df_fob_grouped = df.groupby(dic_group_keys[periodo]).max()
    df_fob_grouped['x'] = df_fob_grouped.index 
    return df_fob_grouped

def criar_figura_matplotlib_agrupado(df, tipos_agrupamento:list, periodo):
    fig, ax = plt.subplots(figsize=(10, 7))
    #dic_tipo = {
    #    'Soma': group_sum(df, periodo),
    #    'Média': group_mean(df, periodo),
    #    'Min': group_min(df, periodo),
    #    'Max': group_max(df, periodo)
    #}
    
    for tipo in tipos_agrupamento:
        if tipo == 'Soma':
            df_fob_grouped = group_sum(df, periodo)
        elif tipo == 'Média':
            df_fob_grouped = group_mean(df, periodo)
        elif tipo == 'Min':
            df_fob_grouped = group_min(df, periodo)
        elif tipo == 'Max':
            df_fob_grouped = group_max(df, periodo)
        #df_fob_grouped = dic_tipo[tipo]
        ax.plot(df_fob_grouped['x'], df_fob_grouped['preco'])
        ax.set_xlabel('Periodo', {'color': 'red'})
        ax.set_ylabel('Preço', {'color': 'red'})
    return fig

# Carrega dataframe com as colunas preco e data

file = open('./df_fob.p', 'rb')
df_fob = pickle.load(file)
file.close()

st.set_page_config(page_title='Analise petroleo', 
                page_icon=':factory:',
                layout="wide", 
                initial_sidebar_state="auto", menu_items=None)

df_fob = apply_model(df_fob)
#st.dataframe(df_fob.head())

st.title('Análise temporal dos dados da INP :oil_drum:')

tab_projeto, tab_fonte = st.tabs(['Projeto', 'Fonte'])

with tab_projeto:
    st.write('''
        O projeto apresentado, usa dados do preço do barril do petrole bruto Brent (FOB) 
        extraídos do site Energy Information Administration (EIA) - http://www.eia.doe.gov 
        para elaborar um modelo da previsão dos dados. Preço por barril do petróleo bruto tipo Brent. 
        Produzido no Mar do Norte (Europa), 
        Brent é uma classe de petróleo bruto que serve como benchmark para o preço internacional 
        de diferentes tipos de petróleo. Neste caso, é valorado no chamado preço FOB (free on board),
        que não inclui despesa de frete e seguro no preço.

        A unidade de medida usada é em dolares (US$)
            
        ***Link da fonte de dados: http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view***
    ''')

    st.write('O dicionario dos dados é o seguinte:')
    st.write({
        'Data': 'Dia da negociação do barril de petroleo bruto',
        'Preço': 'Valor em dolar do barril do petroleo bruto',
        'Ano': 'Valor isolado do ano em questão da negociação',
        'Aumento': 'Variavação do preço do petroleo em relação aos dias de negociação',
        'Aceleração': 'Cálculo de como a variação do preço se multiplica',
        'Dia da semana': 'Valor do dia da semana'
    })

    btn_atualizar = st.button('Atualizar dados')
    if btn_atualizar:
        with st.spinner('Atualizando dados...'):
            df_fob = transform_dataframe_web()

    st.subheader('Modelo da análise')
    st.write("""
    O método CRISP (Cross-Industry Standard Process for Data Mining) é uma abordagem padrão
    para a realização de projetos de mineração de dados.
    Ele proporciona uma estrutura organizada em diversas etapas, incluindo:

    1. **Entendimento do Negócio (Business Understanding):** 
    - Compreender os objetivos de negócio relacionados ao preço do petróleo, 
        como prever tendências, identificar fatores de influência e tomar decisões estratégicas.

    2. **Entendimento dos Dados (Data Understanding):**
    - Explorar e entender a base de dados sobre o preço do petróleo. \
    Isso inclui examinar as variáveis disponíveis, identificar possíveis fontes de dados adicionais e \
        compreender a qualidade e a integridade dos dados.

    3. **Preparação dos Dados (Data Preparation):**
    - Limpar, transformar e preparar os dados para análise.\
        Isso pode envolver o tratamento de valores ausentes, \
        normalização de dados e a criação de novas variáveis relevantes.

    4. **Modelagem (Modeling):**
    - Selecionar e aplicar técnicas de modelagem adequadas \
        para entender as relações entre as variáveis e prever o preço do petróleo. \
        Com isso foi utilizado com séries temporais ou outras técnicas de aprendizado de máquina.

    5. **Avaliação (Evaluation):**
    - Avaliar a qualidade e a eficácia dos modelos desenvolvidos.\
        Isso pode envolver a utilização de métricas de desempenho, \
        validação cruzada e comparação de diferentes modelos.

    6. **Implantação (Deployment):**
    - Implementar os modelos selecionados em ambientes operacionais. Isso pode incluir a integração dos modelos em sistemas de suporte à decisão ou a criação de ferramentas para monitorar continuamente o preço do petróleo.

    Ao utilizar o método CRISP, garantimos uma abordagem sistemática, orientada pelos objetivos de negócio e resultados práticos e acionáveis. 
    """)

    with st.expander('Dashboards', True):
        df_fob.sort_values('data')
        st.write('O período mais recente dos dados disponibilizados é: **' + format_date(df_fob['data'].iloc[1])+ '**')
        st.write('A data do primeiro resgistro apresentado é: **' + format_date(df_fob['data'].iloc[-1]) + '**')
        
        
        col1, col2 = st.columns(2)

        inp_dt_inicio = col1.date_input('Data Inicio', value=df_fob['data'].iloc[-1],
                                        min_value=df_fob['data'].iloc[-1],
                                        max_value=df_fob['data'].iloc[-1])
        
        inp_dt_fim = col2.date_input('Data Fim', value=df_fob['data'].iloc[1],
                                    min_value=df_fob['data'].iloc[1],
                                    max_value=df_fob['data'].iloc[1])

        options_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sabado', 'Domingo']
        inp_days = st.multiselect('Seleção dias semana', options_dias, default=options_dias)

        df_fob = df_fob.loc[
                (df_fob['data'].dt.date >=  inp_dt_inicio)&
                (df_fob['data'].dt.date <= inp_dt_fim) &
                (df_fob['dia_semana'].isin(inp_days)) 
            ]

        chbox_agrupar = st.checkbox('Agrupar')
        
        if chbox_agrupar:
            
            inp_periodo_agrupamento = st.selectbox('Periodo', ['Anual','Mensal', 'Semanal'])
            inp_funcao = st.multiselect('Função', ['Soma', 'Média', 'Min', 'Max'])
            df_fob['WeekNumber'] = df_fob['data'].dt.isocalendar().week
            df_fob['mes'] = df_fob['data'].dt.month

            st.write('Variação do preço agrupado pelo período')
            fig_graph = criar_figura_matplotlib_agrupado(df_fob, inp_funcao, inp_periodo_agrupamento)
            st.plotly_chart(fig_graph)

        else:
            st.markdown(" <h3>Eventos</h3> \
                            <p>Acontecimentos que podem ter tido relação direta com os valores apresentados</p>\
                            <li>Crise Financeira Americana (2008)</li> \
            <li>Primavera Árabe (2011)</li>\
            <li>Pandemia de COVID-19 (2019)</li> \
            <li>Tendência de Energias Renováveis (2020)</li> ", unsafe_allow_html=True)

            st.write('Contagem dos registros: ' + str(df_fob.shape[0]))

            # inp_valores =  st.multiselect('Valores de analise', ['preco', 'aumento','aceleracao'], 'preco')
            # if len(inp_valores) > 0:
            #     st.line_chart(data=df_fob, x='data', y=inp_valores)

            st.write('Variação do preço do petróleo')
            st.plotly_chart(criar_figura_matplotlib(df_fob,'data','preco',tipo_grafico='line',titulo='',
                                                    xlabel='Data', ylabel='Preço'))
            
            st.write('Composição do preço do petróleo')
            st.plotly_chart(criar_figura_matplotlib(df_fob,'data','preco',tipo_grafico='hist',titulo='',
                                                    xlabel='Data', ylabel='Preço'))

    with st.expander('Modelo'):
        st.subheader('Previsão')

        try:
            with st.spinner('Carregando dados mais atualizados para o modelo'):
                ipeadata = pd.read_html(r"http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view", encoding='utf-8', header=0)[2]
                #ipeadata.to_pickle('df_fob.pkl')
                if ipeadate.shape[0] < 0:
                    file = open('./df_fob.pkl', 'rb')
                    ipeadata = pickle.load(file)
                    file.close()
                    
        except Exception:
            file = open('./df_fob.pkl', 'rb')
            ipeadata = pickle.load(file)
            file.close()

        # Tratamento no Dataframe
        ipeadata.columns = ['Data', 'Preco']

        ipeadata['Data'] = pd.to_datetime(ipeadata['Data'], format='%d/%m/%Y')
        ipeadata.Preco = ipeadata.apply(lambda row: correct_value_with_separators(row), axis=1)
        ipeadata.Preco = ipeadata.Preco.astype(float)

        st.write('''Com a seleção de periodo de tempo, é iniciado a predição do valor do preço do petroleo.
                Esse indicativo pode auxiliar a entender qual seria a tendência do mercado, facilitando a tomada de decisão''')

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
            st.write('*DESCRIÇÃO*')
            st.write('MAE - (Mean Absolute Error), **erro médio absoluto** com a métrica é feita a média \
                    da diferença entre o valor real com a previsão do modelo.')
            st.write('MSE - (Mean Squared Error), **erro quadrático médio**, feita da mesma forma \
                    que o MAE, porém ela é elava o quadrado a diferença. Desta forma fica visivel os valores \
                    diferentes. Quanto maior menor a eficiência do modelo.')
            st.write('MAPE - (Mean Absolute Percentual Error), **erro percentual absoluto médio**, mostra a \
                    porcentagem do erro em relação ao valores reais.')
            st.write('R2 Score -representa o percentual da variância dos dados que é explicado pelo modelo. \
                    Os resultados variam de 0 a 1, geralmente também são expressos em termos percentuais,\
                    ou seja, variando entre 0% e 100%. Quanto maior é o valor de R², mais explicativo é o modelo \
                    em relação aos dados previstos. ')
            

            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric('MAE: ', round(metrics[0], 2))
            col2.metric('MSE: ', round(metrics[1], 2))
            col3.metric('MAPE: ', round(metrics[2], 2))
            col4.metric('R2 Score: ', str(round(metrics[3] * 100, 2))+"%")
        
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_and_future.index, y=df_and_future['preco'], mode='lines', name='valores reais'))
            fig.add_trace(go.Scatter(x=df_and_future.index, y=df_and_future['pred'], mode='lines', name='valores preditos'))
            
            st.plotly_chart(fig)
        else:
            pass

with tab_fonte:
    st.subheader('Sobre')
    st.write('Projeto desenvolvido por Estudantes da FIAP - Pós Tech em Data Analytics.')

    st.subheader('Github')
    st.write('https://github.com/gabriel-rosa-tech/st-inp')

    st.subheader('Membros')
    st.markdown("<li>Barbara Campos</li> \
            <li>Brendon Calazans</li>\
            <li>Carlos Eduardo</li> \
            <li>Gabriel Rosa</li>", unsafe_allow_html=True)
