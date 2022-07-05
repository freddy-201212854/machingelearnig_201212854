from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from soupsieve import select
import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from io import StringIO
import pandas as pd

import pip
pip.main(["install", "openpyxl"])

st.title('Machine Learning 20121854')
st.sidebar.title('Panel de opciones')
fp = st.sidebar.file_uploader(
    "Carga de Archivos", type=['csv', 'xls', 'xlsx', 'json', ])

dataset_name = st.sidebar.selectbox(
    'Seleccionar Algoritmo',
    ("Regresión Lineal", 'Regresión Polinomial', 'Clasificador Gaussiano',
     "Clasificador de Árboles de decisión", "Redes Neuronales")
)

if fp is not None:
    st.write(f"## {dataset_name}")
else:
    st.markdown(
        f'<h6 style="color:gray;font-size:14;">{"Seleccione un archivo desde el panel de opciones y realice las operaciones con los algoritmos disponibles."}</h6>', unsafe_allow_html=True)
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
            # st.dataframe(df)
        elif (fp.type == 'application/vnd.ms-excel' or fp.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            df = pd.read_excel(fp)
            # st.dataframe(df)
        elif (fp.type == 'application/json'):
            df = pd.read_json(fp)
            # st.dataframe(df)

        if st.checkbox('Mostrar tabla de datos'):
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

                    st.write(f"Coeficiente: {a1[0]}")
                    st.write(f"Interceptor: {a2}")
                    st.write("y = ", a1[0], "*x +", a2)

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
                st.write(f"Predicción para { data_prdit }:", pred[0])

                a1 = alg.coef_
                a2 = alg.intercept_

                st.write(f"Coeficiente: {a1[0]}")
                st.write(f"Interceptor: {a2}")
                st.write("y = ", a1[0], "*x +", a2)

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

                    st.write('Coeficientes: ', reg.coef_)
                    st.write('Coeficientes = ' + str(reg.coef_) +
                             ', Interceptor = ' + str(reg.intercept_))

                    a1 = reg.coef_
                    a2 = reg.intercept_

                    total = ''
                    contador = 0
                    for x in a1:
                        for y2 in x:
                            total += str(y2) + 'x^'+str(contador) + "+"
                            contador += 1

                    total = total + str(a2[0])
                    st.write(f"y = {total} ")

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

                st.write('Coeficientes:', reg.coef_)
                st.write('Coeficientes = ' + str(reg.coef_) +
                         ', Interceptor = ' + str(reg.intercept_))

                a1 = reg.coef_
                a2 = reg.intercept_

                y = a1 * data_predict + a2

                total2 = ''
                contador2 = 0
                for x2 in a1:
                    for y3 in x2:
                        total2 += str(y3) + 'x^'+str(contador2) + "+"
                        contador2 += 1

                    total2 = total2 + str(a2[0])
                    st.write(f"y = {total2} ")

                total = 0
                contador = 0
                for x in a1:
                    for y2 in x:
                        total += y2 * (data_predict**contador)
                        contador += 1

                total = total + a2
                st.write(f"Predicción para  { data_predict }:", total[0])

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

            valuesPredict = np.array([])
            if classifier_name == 'Predicción de la tendencia':
                for column in chooseColumns:
                    valuesPredict = np.append(
                        valuesPredict, st.text_input("Ingrese el valor de " + column + ": "))

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
                try:
                    classifier = GaussianNB()
                    classifier.fit(X_train, y_train)

                    array_predict = []
                    for x in valuesPredict:
                        array_predict.append(float(x))
                    # st.write(str(array_predict))

                    pred = classifier.predict([array_predict])
                    st.write("Predicción: ", str(pred))
                except:
                    st.write("Seleccione o ingrese los campos necesarios")

        elif (dataset_name == 'Clasificador de Árboles de decisión'):
            options = df.columns
            choose = st.selectbox("Seleccionar columna objetivo", (options))
            chooseColumns = st.multiselect(
                "Seleccionar columnas a clasificar", (df.drop(columns=[choose]).columns))

            calculatePredict = st.radio(
                'Realizar Parametrización', ('No', 'Si'))
            valuesPredict = np.array([])
            if calculatePredict == 'Si':
                for column in chooseColumns:
                    valuesPredict = np.append(
                        valuesPredict, st.text_input("Ingrese el valor de " + column + ": "))

            X = df[chooseColumns]
            y = df[choose]

            y = df[choose].values
            le = LabelEncoder()
            y = le.fit_transform(y.flatten())

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=0)
            except Exception as e:
                st.write(
                    'Ha ocurrido un error al clasificar el algoritmo de árboles de decisión', e)

            clf = DecisionTreeClassifier(
                criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write("Precisión del modelo:")
            st.write(accuracy_score(y_test, y_pred))
            if calculatePredict == 'Si':
                valuesPredict = valuesPredict.reshape(1, -1)
                pred = clf.predict(valuesPredict)
                st.write("Predicción: ",
                         str(le.inverse_transform(pred)[0]))
            dot_data = tree.export_graphviz(clf, out_file=None)
            st.graphviz_chart(dot_data)

        elif (dataset_name == 'Redes Neuronales'):
            options = df.columns
            choose = st.selectbox("Seleccionar columna objetivo", (options))
            chooseColumns = st.multiselect(
                "Seleccionar columnas a clasificar", (df.drop(columns=[choose]).columns))

            calculatePredict = st.radio(
                'Realizar Parametrización', ('No', 'Si'))
            valuesPredict = np.array([])
            if calculatePredict == 'Si':
                for column in chooseColumns:
                    valuesPredict = np.append(
                        valuesPredict, st.text_input("Ingrese el valor de " + column + ": "))

            X = df[chooseColumns]
            y = df[choose]

            y = df[choose].values
            le = LabelEncoder()
            y = le.fit_transform(y.flatten())

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=0)
            except Exception as e:
                st.write(
                    'Ha ocurrido un error al clasificar el algoritmo de árboles de decisión', e)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            # Instantiate the Classifier and fit the model.
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred) * 100
            report = classification_report(y_test, y_pred)
            st.text("Precisión del modelo: ")
            st.write(score, "%")
            st.text("Reporte del modelo: ")
            st.write(report)

    except Exception as e:
        st.write('Ha ocurrido un error con el archivo cargado', e)
