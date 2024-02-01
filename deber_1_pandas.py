#Importamos la libreria pandas
import pandas as pd

#Variable de texto 
Variabletext= "HOLA"
print(Variabletext)

##Lista de numeros :::::::::::::::::::::::::::::::::
lista_1= [1,2,3,4,5,6]
print(lista_1)

#Dicconario de algunos grupos de alimentos::::::::::::::::
diccionario_1={"fruras":1,"verduras":2,"carnes":3,"postres":4}
print(diccionario_1)

# Vectores con 8 elementos repetidos cada uno
vector_entero = [5] * 8
vector_flotante = [2.72] * 8 #(Tiene decimales)
vector_complejo = [(6 + 3j)] * 8

#Diccionario que contenga los vertores definidos antes:
dicvectores = {
    "entero": vector_entero,
    "flotante": vector_flotante,
    "complejo": vector_complejo
}

print(dicvectores)

##Tuplas:Son inmutables, lo que significa que no pueden ser modificadas una vez declaradas, y en vez de inicializarse con corchetes se hace con ()
tupla=(3,5,7,9)
print(tupla)

##Cadenas
cadena_simple = 'Mi nomre es Joselyn'

cadena_doble = ["Me gusta bailar", "Me gusta aprender"]

print(cadena_simple)
print(cadena_doble)


##Lectura de data frame usando Pandas
#importar datos
imp_sri=pd.read_excel("ventas_SRI.xlsx")
print(imp_sri)
#Lectura de tipos de datos 
imp_sri.dtypes
imp_sri.columns
imp_sri.describe()##Estadistica descriptiva de cada una d elas variables de la data.