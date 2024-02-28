#TRABAJO FINAL
#Nombre: Joselyn Allauca

#*******************************************1er paso*****************************

##Importar librerias

import pandas as pd
import numpy as np
import sklearn
# divicion de  conjuntos de datos y validación cruzada
from sklearn.model_selection import train_test_split, KFold
#Pentrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#Análisis estadísticos y estimación de modelos.
import statsmodels.api as sm
#Graficos de datos
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
#***********************Variable asignada y Poblacion
#Variable: categoria_seguridad_alimentaria
#Poblacion Mujer

##***********************************2do paso****************************************
# Cargar los datos
datos = df = pd.read_csv("data.txt", sep=";")
datos
print(datos.columns)
##*****************Eliminamos datos finitos o faltantes
print(datos.isna().sum())
variables=['sexo','etnia','dcronica','region', 'n_hijos', 'tipo_de_piso',
       'espacio_lavado', 'categoria_seguridad_alimentaria', 'condicion_empleo',
       'quintil', 'categoria_cocina', 'categoria_agua', 'serv_hig',
       'fexp_nino']
datos_limpios = datos.dropna(subset=variables)
#Comprobamos si hay datos faltanates:
print(datos_limpios.isna().sum())
for i in variables:
    datos = datos[~datos[i].isna()]
print(datos.isna().sum())
#*****************************Pregunta 1 ****************************************************
# Filtrar datos para la población objetivo (niños en la región Sierra)
pob = datos_limpios[(datos_limpios ["sexo"] == "Mujer") & (datos_limpios ["categoria_seguridad_alimentaria"] )]
print(pob.columns)
# Calcular cuántos niños se encuentran en la población objetivo
len(pob)
pob["categoria_seguridad_alimentaria"].describe()
datos.groupby("categoria_seguridad_alimentaria").size()
#*************************** Pregunta 2**********************************************************
datos_limpios["sa_bi"] = datos_limpios["categoria_seguridad_alimentaria"].apply(lambda x:1 if x== "Inseguridad alta " else 0)

#Separamos varaibles categoricas y numericas (las mas relevantes)
variables_categoricas = ['region', 'sexo', 'condicion_empleo']
variables_numericas = ['n_hijos']

#Crear un transformador para estandarizar ariables numericas 
transformador = StandardScaler()
datos_escalados = datos_limpios.copy()
print(datos_escalados.isna().sum())

#Estandarizacion de la varaible nnumerica
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])
#Conversion de variables categoricas a Dummy
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
print(datos_dummies.isna().sum())
#Seleccion variables independientes y varaible dependiente para el modelo
X = datos_dummies[['n_hijos', 'region_Sierra', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = datos_dummies["sa_bi"]
weights = datos_dummies['fexp_nino']
print(weights.isna().sum())
#Separación de las muestras en entramiento (train) y prubea (test)
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)
# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables1 = X_train.columns

for i in variables1:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)
    y_train = y_train.astype(int)

modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#Respuesta:
#¿Cuál es el valor del parámetro asociado a la variable clave si ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?
## el valor del parámetro asociado a la variable n_hijos, sería -13.1717 no es significativo.

#Interpretacion
#Para todos los parametros correspondientes a las variables explicativas del modelo tenemos qe ninguni es significativo por 
#lo tanto niguna de estas variables explica adecuadamente la seguridad alimentaria 


#******************************************Pregunta 3 ***************************************************
# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)

# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
comparacion = (predictions_class == y_test)

# Definir el número de folds para la validación cruzada
kf = KFold(n_splits=100)

accuracy_scores = []  # Lista para almacenar los puntajes de precisión de cada fold
df_params = pd.DataFrame()  # DataFrame para almacenar los coeficientes estimados en cada fold

# Iterar sobre cada fold
for train_index, test_index in kf.split(X_train):
    
    # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustar un modelo de regresión logística en el conjunto de entrenamiento de este fold
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraer los coeficientes y organizarlos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizar predicciones en el conjunto de prueba de este fold
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calcular la precisión del modelo en el conjunto de prueba de este fold
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenar los coeficientes estimados en este fold en el DataFrame principal
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

# Calcular la precisión promedio de la validación cruzada
mean_accuracy = np.mean(accuracy_scores)
print(f"Precisión promedio de validación cruzada: {mean_accuracy}")

# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

# Crear el histograma
plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio - 0.1, plt.ylim()[1] - 0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar el título y etiquetas de los ejes
plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

# Crear el histograma de los coeficientes para la variable "n_hijos"
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Calcular la media de los coeficientes para la variable "n_hijos"
media_coeficientes_n_hijos = np.mean(df_params["n_hijos"])

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(media_coeficientes_n_hijos, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(media_coeficientes_n_hijos - 0.1, plt.ylim()[1] - 0.1, f'Media de los coeficientes: {media_coeficientes_n_hijos:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

# Configurar título y etiquetas de los ejes
plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()
