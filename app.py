import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from pickle import load

# Load the pre-trained model and scaler from local files
model_path = "clfLMBest.pkl"
scaler_path = "StandardScaler.pkl"

model = joblib.load(model_path)
scaler = load(open(scaler_path, 'rb'))

# Define columns to encode
categorical_columns = ['job', 'marital', 'default', 'month', 'day_of_week', 'campaign', 'pdays', 'poutcome']
numerical_columns = ['age', 'education', 'housing', 'loan', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Define categories for OneHotEncoder
categories = [
    ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
     'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'],
    ['divorced', 'married', 'single'],
    ['no', 'unknown', 'yes'],
    ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'],
    ['fri', 'mon', 'thu', 'tue', 'wed'],
    ['1 contacto', '2 contactos', '3 contactos', 'Mais que 3 contactos'],
    ['Há mais de uma semana', 'Na ultima semana', 'Nunca foi contactado'],
    ['failure', 'nonexistent', 'success']
]

# Define mappings for Portuguese options
job_mapping = {
    'Administrador': 'admin.',
    'Operário': 'blue-collar',
    'Empreendedor': 'entrepreneur',
    'Empregada Doméstica': 'housemaid',
    'Gestor': 'management',
    'Reformado': 'retired',
    'Trabalhador por Conta Própria': 'self-employed',
    'Serviços': 'services',
    'Estudante': 'student',
    'Técnico': 'technician',
    'Desempregado': 'unemployed'
}

marital_mapping = {
    'Divorciado': 'divorced',
    'Casado': 'married',
    'Solteiro': 'single'
}

education_mapping = {
    'Analfabeto': 'illiterate',
    '4º ano': 'basic.4y',
    '6º ano': 'basic.6y',
    '9º ano': 'basic.9y',
    'Ensino Secundário': 'high.school',
    'Curso Profissional': 'professional.course',
    'Ensino Superior': 'university.degree'
}

contact_mapping = {
    'Celular': 'cellular',
    'Telefone': 'telephone'
}

poutcome_mapping = {
    'Fracasso': 'failure',
    'Inexistente': 'nonexistent',
    'Sucesso': 'success'
}

# Define preprocessing steps
def preprocess_data(df):
    # Map categorical values
    df["cellular"] = df.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8')
    df = df.drop(columns='contact')

    # Binning campaign values
    bins = [0, 1, 2, 3, float('inf')]
    labels = ["1 contacto", "2 contactos", "3 contactos", "Mais que 3 contactos"]
    df['campaign'] = pd.cut(df['campaign'], bins=bins, labels=labels, right=True, duplicates="drop")

    # Replace and bin pdays values
    df.pdays = df.pdays.replace({999: -1})
    bins = [-float('inf'), 0, 7, float('inf')]
    labels = ['Nunca foi contactado', 'Na ultima semana', 'Há mais de uma semana']
    df['pdays'] = pd.cut(df['pdays'], bins=bins, labels=labels, right=False, duplicates="drop")

    # Map housing and loan values
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0}).astype('uint8')
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0}).astype('uint8')

    # Map education values
    df["education"] = df["education"].replace({
        'illiterate': 0,
        'basic.4y': 1,
        'basic.6y': 2,
        'basic.9y': 3,
        'high.school': 4,
        'professional.course': 5,
        'university.degree': 6
    }).astype('uint8')

    # Bin age values
    bins = [0, 24, 34, 44, 54, 64, float('inf')]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    df['age'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['age'] = df['age'].cat.rename_categories({
        '<25': 1,
        '25-34': 2,
        '35-44': 3,
        '45-54': 4,
        '55-64': 5,
        '65+': 6
    }).astype('uint8')

    # Map previous values
    df['previous'] = df.apply(lambda row: 0 if row['previous'] == 0 else 1, axis=1)

    return df

# App title and layout
st.set_page_config(page_title='Formulário de Previsão de Depósitos Bancários', page_icon='📈', layout='wide')
st.title('Formulário de Previsão de Depósitos Bancários')

st.image('https://www.ipleiria.pt/wp-content/themes/ipleiria/img/logo_ipl_header.png', width=200)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Formulário", "Sobre"],
    icons=["pencil-square", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Introductory message
st.markdown("""
Bem-vindo ao formulário de previsão de depósito bancário. Preencha as informações abaixo para ajudar a prever se um cliente irá subscrever um depósito a prazo.
""")

# Define options for form fields
job_options = list(job_mapping.keys())
marital_options = list(marital_mapping.keys())
default_options = ['Não', 'Desconhecido', 'Sim']
month_options = ['abril', 'agosto', 'dezembro', 'julho', 'junho', 'março', 'maio', 'novembro', 'outubro', 'setembro']
day_of_week_options = ['sexta-feira', 'segunda-feira', 'quinta-feira', 'terça-feira', 'quarta-feira']
education_options = list(education_mapping.keys())
contact_options = list(contact_mapping.keys())
poutcome_options = list(poutcome_mapping.keys())

# Form fields
if selected == "Formulário":
    with st.form(key='bank_form'):
        st.header('Informações Pessoais')
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Idade', min_value=18, max_value=100)
            job = st.selectbox('Trabalho', job_options)
            marital = st.selectbox('Estado Civil', marital_options)
            education = st.selectbox('Educação', education_options)

        with col2:
            default = st.selectbox('Crédito em Default', default_options)
            housing = st.selectbox('Empréstimo Habitacional', ['Sim', 'Não'])
            loan = st.selectbox('Empréstimo Pessoal', ['Sim', 'Não'])

        st.header('Detalhes do Contacto')
        col3, col4 = st.columns(2)

        with col3:
            contact = st.selectbox('Tipo de Contacto', contact_options)
            month = st.selectbox('Mês de Contacto', month_options)
            day_of_week = st.selectbox('Dia da Semana de Contacto', day_of_week_options)
            poutcome = st.selectbox('Resultado da Campanha Anterior', poutcome_options)

        with col4:
            campaign = st.number_input('Número de Contactos na Campanha', min_value=0)
            pdays = st.number_input('Dias desde o Último Contacto na Campanha', min_value=0, max_value=999)
            previous = st.number_input('Número de Contactos Anteriores', min_value=0)

        st.header('Indicadores Económicos')
        col5, col6 = st.columns(2)

        with col5:
            emp_var_rate = st.number_input('Taxa de Variação do Emprego')
            cons_price_idx = st.number_input('Índice de Preços ao Consumidor')
            cons_conf_idx = st.number_input('Índice de Confiança do Consumidor')

        with col6:
            euribor3m = st.number_input('Taxa EURIBOR a 3 Meses')
            nr_employed = st.number_input('Número de Empregados', min_value=0)

        # Submit button
        submit_button = st.form_submit_button(label='Submeter')

    if submit_button:
        if model is not None and scaler is not None:

            original_input_data = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'housing': housing,
                'loan': loan,
                'month': month,
                'day_of_week': day_of_week,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'poutcome': poutcome,
                'emp.var.rate': emp_var_rate,
                'cons.price.idx': cons_price_idx,
                'cons.conf.idx': cons_conf_idx,
                'euribor3m': euribor3m,
                'nr.employed': nr_employed,
                'contact': contact
            }

            # Collect data from form
            input_data = pd.DataFrame([{
                'age': age,
                'job': job_mapping[job],
                'marital': marital_mapping[marital],
                'education': education_mapping[education],
                'default': default.lower(),
                'housing': 'yes' if housing.lower() == 'sim' else 'no',
                'loan': 'yes' if loan.lower() == 'sim' else 'no',
                'month': month[:3],
                'day_of_week': day_of_week[:3],
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'poutcome': poutcome_mapping[poutcome],
                'emp.var.rate': emp_var_rate,
                'cons.price.idx': cons_price_idx,
                'cons.conf.idx': cons_conf_idx,
                'euribor3m': euribor3m,
                'nr.employed': nr_employed,
                'contact': contact_mapping[contact]
            }])

            # Preprocess input data
            input_data_transformed = preprocess_data(input_data)

            # Transform the entire data
            ct = ColumnTransformer(transformers=[
                ('encoder', OneHotEncoder(sparse_output=False, categories=categories, handle_unknown='ignore'), categorical_columns)
            ], remainder='passthrough')

            final_data = ct.fit_transform(input_data_transformed)

            # Apply StandardScaler
            scaled_data = scaler.transform(final_data)

            # Perform the prediction
            prediction = model.predict(scaled_data)

            st.success('Formulário submetido com sucesso!')
            st.write('### Valores fornecidos:')
            st.write(pd.DataFrame([original_input_data]))

            # Display the prediction result
            if prediction == 1:
                st.write("Com base nos valores fornecidos, o cliente tem boas chances de subscrever um depósito a prazo. 😊")
            else:
                st.write("Com base nos valores fornecidos, o cliente provavelmente não irá subscrever um depósito a prazo. 😢")
        else:
            st.error("Por favor, carregue o modelo e o StandardScaler corretamente.")

# About section
elif selected == "Sobre":
    st.markdown("""
    ### Sobre este Formulário
    Este formulário foi criado para ajudar na previsão de depósitos bancários, com base em dados históricos de campanhas de marketing.
    """)