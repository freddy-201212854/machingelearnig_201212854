from math import degrees
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3
from sklearn.metrics import mean_squared_error, r2_score
import time

import pip
pip.main(["install", "openpyxl"])

st.title('Machine Learning 20121854')
fp = st.sidebar.file_uploader(
    "Carga de Archivos", type=['csv', 'xls', 'xlsx', 'json', ])

dataset_name = st.sidebar.selectbox(
    'Seleccionar Algoritmo',
    ("Regresión Lineal", 'Regresión Polinomial', 'Clasificador Gaussiano',
     "Clasificador de Árboles de decisión", "Redes Neuronales")
)

st.write(f"## {dataset_name}")

classifier_name = st.sidebar.selectbox(
    'Seleccionar Operación',
    ('Graficar Puntos', 'Definir función de tendencia', 'Predicción de la tendencia')
    # 'Clasificar por Gauss', 'Clasificar árboles de decisión', 'Clasificar redes neuronales')
)

if fp is not None:
    try:
        ParamsX = ""
        ParamsY = ""
        df = ""
        file_detail = {"file": fp.name,
                       "filetype": fp.type, "filesize": fp.size}
        st.write(file_detail)
        if (fp.type == 'text/csv'):
            df = pd.read_csv(fp)
            st.dataframe(df)
        elif (fp.type == 'application/vnd.ms-excel' or fp.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            df = pd.read_excel(fp)
            st.dataframe(df)
        elif (fp.type == 'application/json'):
            df = pd.read_json(fp)
            st.dataframe(df)

        if (dataset_name == 'Regresión Lineal'):

            if (classifier_name == 'Graficar Puntos' or classifier_name == 'Definir función de tendencia'):
                ParamsY = st.selectbox('Variable Y:', df.columns)
                ParamsX = st.selectbox(
                    'Variable X:', df.drop(columns=[ParamsY]).columns)

                if (classifier_name == 'Definir función de tendencia'):
                    alg = LinearRegression()
                    X = np.expand_dims(df[ParamsX], 1)
                    y = df[ParamsY]
                    alg.fit(X, y)

                    a1 = alg.coef_
                    a2 = alg.intercept_

                    st.write(a1, "x + ", a2)

                    fig = plt.figure(figsize=(3, 3))
                    plt.scatter(x=df[ParamsX], y=df[ParamsY])
                    x = df[ParamsX]
                    y = a1 * x + a2
                    plt.plot(x, y, 'r')
                    plt.xlabel(ParamsX, fontsize=8)
                    plt.ylabel(ParamsY, fontsize=8)

                    plt.legend(fontsize=6)
                    plt.tick_params(labelsize=6)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.scatter(x=df[ParamsX], y=df[ParamsY])
                    plt.xlabel(ParamsX)
                    plt.ylabel(ParamsY)
                    st.pyplot(fig)
            elif (classifier_name == 'Predicción de la tendencia'):
                ParamsY = st.selectbox('Variable Y:', df.columns)
                ParamsX = st.selectbox(
                    'Variable X:', df.drop(columns=[ParamsY]).columns)
                data_prdit = st.number_input("Ingrese el valor a predecir")

                alg = LinearRegression()
                X = np.expand_dims(df[ParamsX], 1)
                y = df[ParamsY]
                alg.fit(X, y)

                pred = alg.predict(np.array([data_prdit]).reshape(1, 1))
                st.write(f"Predicción para { data_prdit }: {pred}")

                a1 = alg.coef_
                a2 = alg.intercept_

                st.write(a1, "x + ", a2)

                fig = plt.figure(figsize=(3, 3))
                plt.scatter(x=df[ParamsX], y=df[ParamsY])
                x = df[ParamsX]
                y = a1 * x + a2
                plt.plot(x, y, 'r')
                plt.xlabel(ParamsX, fontsize=8)
                plt.ylabel(ParamsY, fontsize=8)

                plt.legend(fontsize=6)
                plt.tick_params(labelsize=6)
                st.pyplot(fig)
        elif (dataset_name == 'Regresión Polinomial'):

            split_data = st.number_input(
                "Ingrese el número de datos en porcentaje a tomar", 1, 100, 100)
            grados = st.number_input(
                "Grados", 1, 6, 2)
            ParamsY = st.selectbox('Variable Y:', df.columns)
            ParamsX = st.selectbox(
                'Variable X:', df.drop(columns=[ParamsY]).columns)
            if (classifier_name == 'Graficar Puntos' or classifier_name == 'Definir función de tendencia'):

                if (classifier_name == 'Definir función de tendencia'):
                    # extracts features from the self.data
                    X = df.iloc[:, 1:-1].values
                    # extracts the labels from the self.data
                    y = df.iloc[:, -1].values
                    # st.write(X)
                    # st.write(y)

                    X = df[ParamsX][:, np.newaxis]
                    Y = df[ParamsY][:, np.newaxis]

                    # degree ile derecesini belirliyoruz.
                    polynomial_regression = PolynomialFeatures(degree=grados)
                    x_polynomial = polynomial_regression.fit_transform(X)

                    # Yapay zekayı Eğitme İşlemi
                    reg = LinearRegression(
                        copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
                    reg.fit(x_polynomial, Y)

                    # Step 4: calculate bias and variance
                    Y_NEW = reg.predict(x_polynomial)

                    rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
                    r2 = r2_score(Y, Y_NEW)

                    st.write('RMSE: ', rmse)
                    st.write('R2: ', r2)

                    st.write(reg.coef_)
                    st.write('w = ' + str(reg.coef_) +
                             ', b = ' + str(reg.intercept_))

                    y_head = reg.predict(x_polynomial)

                    fig = plt.figure(figsize=(3, 3))
                    plt.scatter(X, Y, color="blue")
                    plt.plot(X, y_head, color='red',
                             label='Regresión Polinomial')

                    plt.xlabel(ParamsX, fontsize=8)
                    plt.ylabel(ParamsY, fontsize=8)

                    plt.legend(fontsize=6)
                    plt.tick_params(labelsize=6)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.scatter(x=df[ParamsX], y=df[ParamsY])
                    plt.xlabel(ParamsX)
                    plt.ylabel(ParamsY)
                    st.pyplot(fig)
            elif (classifier_name == 'Predicción de la tendencia'):
                data_predict = st.number_input(
                    "Ingrese valor a predecir")

                # extracts features from the self.data
                X = df.iloc[:, 1:-1].values
                # extracts the labels from the self.data
                Y = df.iloc[:, -1].values
                # st.write(X)
                # st.write(y)

                X = df[ParamsX][:, np.newaxis]
                Y = df[ParamsY][:, np.newaxis]

                # degree ile derecesini belirliyoruz.
                polynomial_regression = PolynomialFeatures(degree=grados)
                x_polynomial = polynomial_regression.fit_transform(X)

                # Yapay zekayı Eğitme İşlemi
                reg = LinearRegression(
                    copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
                reg.fit(x_polynomial, Y)

                # Step 4: calculate bias and variance
                Y_NEW = reg.predict(x_polynomial)

                rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
                r2 = r2_score(Y, Y_NEW)

                st.write('RMSE: ', rmse)
                st.write('R2: ', r2)

                st.write(reg.coef_)
                st.write('w = ' + str(reg.coef_) +
                         ', b = ' + str(reg.intercept_))

                a1 = reg.coef_
                a2 = reg.intercept_

                y = a1 * data_predict + a2

                total = 0
                contador = 0
                for x in a1:
                    for y2 in x:
                        total += y2 * (data_predict**contador)
                        contador += 1

                total = total + a2
                st.write(f"Predicción para  { data_predict }: {total} ")

                y_head = reg.predict(x_polynomial)

                fig = plt.figure(figsize=(3, 3))
                plt.scatter(X, Y, color="blue")
                plt.plot(X, y_head, color='red',
                         label='Regresión Polinomial')

                plt.xlabel(ParamsX, fontsize=8)
                plt.ylabel(ParamsY, fontsize=8)

                plt.legend(fontsize=6)
                plt.tick_params(labelsize=6)
                st.pyplot(fig)

        elif (dataset_name == 'Clasificador Gaussiano'):
            options = df.columns
            choose = st.selectbox("Seleccionar columna objetivo", (options))
            chooseColumns = st.multiselect(
                "Seleccionar columnas a clasificar", (df.drop(columns=[choose]).columns))

            X = df[chooseColumns]
            y = df[choose]
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=0)
            except Exception as e:
                st.write(
                    'Ha ocurrido un error al clasificar el algoritmo Gaussiano', e)

            if (classifier_name == 'Graficar Puntos'):

                standar = StandardScaler()
                X_train = standar.fit_transform(X_train)
                X_test = standar.transform(X_test)

                classifier = GaussianNB()
                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                fig2 = plt.figure(figsize=(3, 3))
                X_set, y_set = X_train, y_train
                X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

                plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

                plt.xlim(X1.min(), X1.max())
                plt.ylim(X2.min(), X2.max())
                for i, j in enumerate(np.unique(y_set)):
                    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                                c=ListedColormap(('red', 'blue'))(i), label=j)
                plt.title('Clasificador Gaussiano')
                plt.xlabel(chooseColumns[0])
                plt.ylabel(chooseColumns[1])
                plt.legend()
                plt.show()
                st.pyplot(fig2)
            elif(classifier_name == 'Predicción de la tendencia'):
                data_predict = st.number_input(
                    "Ingrese valor a predecir")

                classifier = GaussianNB()
                classifier.fit(X_train, y_train)
                XX = [[data_predict]]
                st.write(classifier.predict(XX))

    except Exception as e:
        st.write('Ha ocurrido un error con el archivo cargado', e)
