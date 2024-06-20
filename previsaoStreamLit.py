import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
from pickle import load
import sklearn
predict_row = pd.DataFrame([['admin.', 'divorced', 'yes','jun', 'mon', 0, 31, 'nonexistent', 1, 'illiterate', 'yes', 'yes',
                            4, -1.8, 0, -46.2, 1.365 , 6000, 'telephone']],
                 columns = ['job', 'marital', 'default','month', 'day_of_week', 'campaign', 'pdays', 'poutcome', 'age', 'education', 'housing', 'loan',
                            'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
                            'nr.employed','contact'])


# job
job = ['admin.', 'blue-collar', 'entrepeneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'umemployed']

# marital
marital = ['divorced', 'married', 'single']

# default
default = ['no', 'unknown', 'yes']

# month
month = ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep']

# day of week
day_of_week = ['fri', 'mon', 'thu', 'tue', 'wed']

# campaign
campaign = [
    "1 contacto",
    "2 contactos",
    "3 contactos",
    "Mais que 3 contactos"
]

# pdays
pdays = ['Há mais de uma semana', 'Na ultima semana', 'Nunca foi contactado']

#poutcome
poutcome = ['failure','nonexistent', 'success',]

categories = [job, marital,default, month, day_of_week, campaign, pdays, poutcome]


predict_row["cellular"] = predict_row.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8')
predict_row = predict_row.drop(columns='contact')

####

bins = [0, 1, 2, 3, float('inf')]
labels = campaign
predict_row['campaign'] = pd.cut(predict_row['campaign'], bins=bins, labels=labels, right=True, duplicates  = "drop")


####
predict_row.pdays = predict_row.pdays.replace({999: -1})

bins = [-float('inf'), 0, 7, float('inf')]
labels = pdays
predict_row['pdays'] = pd.cut(predict_row['pdays'], bins=bins, labels=labels, right=False, duplicates  = "drop")

####

predict_row.housing = predict_row.housing.map({'yes': 1, 'no': 0}).astype('uint8')
predict_row.loan = predict_row.loan.map({'yes': 1, 'no': 0}).astype('uint8')

predict_row["education"] = predict_row["education"].replace({
    'illiterate': 0,
    'basic.4y': 1, 
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4, 
    'professional.course': 5, 
    'university.degree': 6})

bins = [0, 24, 34, 44, 54, 64, float('inf')]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']


predict_row['age'] = pd.cut(predict_row['age'], bins=bins, labels=labels, right=False)
predict_row['age'] = predict_row['age'].replace({
    '<25': 1,
    '25-34' : 2,
    '35-44': 3,
    '45-54': 4,
    '55-64':5,
    '65+' : 6
}).astype('uint8')

predict_row['previous'] = predict_row.apply(
    lambda row: 0 if row['previous'] == 0 else 1,
    axis = 1
)


categorical_columns = predict_row.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_columns)
categorical_columns = predict_row.columns.get_indexer(categorical_columns)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False, categories=categories, handle_unknown = 'ignore'), categorical_columns)], remainder='passthrough')

X = np.array(ct.fit_transform(predict_row))


from pickle import load
sc = load(open('StandardScaler.pkl', 'rb'))
print(sc)

X = sc.transform(X)
print(X)

import joblib
clf = joblib.load("clfLMBest.pkl")

prediction = clf.predict(X)[0]

print(prediction)