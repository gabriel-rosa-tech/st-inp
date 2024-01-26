import streamlit as st
import datetime
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

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
    dfs_fob = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view')
    df_fob = dfs_fob[2]
    df_fob.columns = ['data', 'preco']
    df_fob.drop(index=0, inplace=True)
    df_fob['preco'] = df_fob['preco'].apply(conv_float)
    df_fob['data'] = pd.to_datetime(df_fob['data'], format='%d/%m/%Y')
    save_pickle(df_fob)

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
    fig, ax = plt.subplots(figsize=(10, 7))
    eventos = {
        'Crise Financeira de 2008': '2008-01-01',
        'Primavera Árabe': '2011-01-01',
        'Pandemia de COVID-19': '2020-01-01',
        'Tendência de Energias Renováveis': '2020-01-01'  # Supondo que a tendência começou em 2020
    }
    # Adicione condições para outros tipos de gráficos, se necessário
    if tipo_grafico == 'scatter':
        ax.scatter(dataframe[x_col], dataframe[y_col])
    elif tipo_grafico == 'bar':
        ax.bar(dataframe[x_col], dataframe[y_col])
    elif tipo_grafico == 'line':
        ax.plot(dataframe[x_col], dataframe[y_col],color='red')
        for evento, data in eventos.items():
            ax.axvline(pd.to_datetime(data), linestyle='--', label=evento)
            ax.legend()
    # Adicione mais opções conforme necessário
    ax.grid()
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig

def group_sum(df, periodo):
    df_fob_grouped = df.groupby(dic_group_keys[periodo]).sum()
    df_fob_grouped['x'] = df_fob_grouped.index 
    return df_fob_grouped

def group_mean(df, periodo):
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
    dic_tipo = {
        'Soma': group_sum(df, periodo),
        'Média': group_mean(df, periodo),
        'Min': group_min(df, periodo),
        'Max': group_max(df, periodo)
    }
    
    for tipo in tipos_agrupamento:
        df_fob_grouped = dic_tipo[tipo]
        ax.plot(df_fob_grouped['x'], df_fob_grouped['preco'])
    
    return fig

# Carrega dataframe com as colunas preco e data
file = open('./df_fob.p', 'rb')
df_fob = pickle.load(file)
file.close()

st.set_page_config(page_title='Analise petroleo', 
                page_icon='random',
                layout="wide", 
                initial_sidebar_state="auto", menu_items=None)

df_fob = apply_model(df_fob)
st.dataframe(df_fob.head())

st.title('Análise temporal dos dados da INP')

st.write('''
    O projeto apresentado, usa dados do preço do barril do petrole bruto Brent (FOB) 
    extraídos do site Energy Information Administration (EIA) - http://www.eia.doe.gov 
    para elaborar um modelo da previsão dos dados. Preço por barril do petróleo bruto tipo Brent. 
    Produzido no Mar do Norte (Europa), 
    Brent é uma classe de petróleo bruto que serve como benchmark para o preço internacional 
    de diferentes tipos de petróleo. Neste caso, é valorado no chamado preço FOB (free on board),
    que não inclui despesa de frete e seguro no preço.

    A unidade de medida usada é em dolares (US$)
        
    ***Link da fonte de dados: http://www.ipeadata.gov.br/ExibeSerie.aspx***
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
        transform_dataframe_web()

st.subheader('Modelo da análise')
st.write("""
O método CRISP (Cross-Industry Standard Process for Data Mining) é uma abordagem padrão
para a realização de projetos de mineração de dados.
Ele proporciona uma estrutura organizada em diversas etapas, incluindo:

1. **Entendimento do Negócio (Business Understanding):** - Compreender os objetivos de negócio relacionados ao preço do petróleo, 
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

    inp_dt_inicio = col1.date_input('Data Inicio', value=datetime.date(1985, 1, 1))
    inp_dt_fim = col2.date_input('Data Inicio', value=datetime.date(2023, 1, 1))

    options_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sabado', 'Domingo']
    inp_days = st.multiselect('Seleção dias semana', options_dias, default=options_dias)

    chbox_agrupar = st.checkbox('Agrupar')
    
    if chbox_agrupar:
        dic_group_keys = {
            'Anual': 'ano',
            'Mensal': 'mes',
            'Semanal': 'WeekNumber'
        }
        inp_periodo_agrupamento = st.selectbox('Periodo', ['Anual','Mensal', 'Semanal'])
        inp_funcao = st.multiselect('Função', ['Soma', 'Média', 'Min', 'Max'])
        df_fob['WeekNumber'] = df_fob['data'].dt.isocalendar().week

        fig_graph = criar_figura_matplotlib_agrupado(df_fob, inp_funcao, inp_periodo_agrupamento)
        st.plotly_chart(fig_graph)
        # if inp_funcao == 'Soma':
        #     df_fob_grouped = df_fob.groupby(dic_group_keys[inp_periodo_agrupamento]).sum()
        #     df_fob_grouped['x'] = df_fob_grouped.index 

        # elif inp_funcao == 'Média':
        #     df_fob_grouped = df_fob.groupby(dic_group_keys[inp_periodo_agrupamento]).mean()
        #     df_fob_grouped['x'] = df_fob_grouped.index 

        # elif inp_funcao == 'Min':
        #     df_fob_grouped = df_fob.groupby(dic_group_keys[inp_periodo_agrupamento]).min()
        #     df_fob_grouped['x'] = df_fob_grouped.index 

        # elif inp_funcao == 'Max':
        #     df_fob_grouped = df_fob.groupby(dic_group_keys[inp_periodo_agrupamento]).max()
        #     df_fob_grouped['x'] = df_fob_grouped.index 
        
        # st.line_chart(data=df_fob_grouped, x='x', y='preco')

    else:
        df_fob = df_fob.loc[
            (df_fob['data'].dt.date >=  inp_dt_inicio)&
            (df_fob['data'].dt.date <= inp_dt_fim) &
            (df_fob['dia_semana'].isin(inp_days)) 
        ]

        st.write('Contagem dos registros: ' + str(df_fob.shape[0]))
        st.line_chart(data=df_fob, x='data', y='preco')

        st.plotly_chart(criar_figura_matplotlib(df_fob,'data','preco',tipo_grafico='line',titulo='Variação do preço do petroleo',
                                                xlabel='Data', ylabel='Preço'))

with st.expander('Modelo'):
    st.subheader('Previsão')
