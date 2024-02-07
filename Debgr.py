#Integrantes:
#Joselyn Allauca
#Maritza Pilco
#María Ruano
# Este será mi script para visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns


#Importamos nuestras 2 bases de datos.
data = pd.read_excel("INTER.xls",sheet_name="Metadata - Countries")
data_2 = pd.read_excel("INTER.xls",sheet_name="Data", skiprows= 3)

#*************************1ra Actividad********************************************************
#Filtrar la base de datos para paises de america latina y para el año de 2020

#Filtramos los paises por la  region America Latina
#de la base de datos de la pestaña Metadata - Countries
inter_lt=data[data["Region"]=="América Latina y el Caribe (excluido altos ingresos)"]["Country Name"]
print(inter_lt)
type(inter_lt)

#Filtramos la segunda  data segun los paises que de America Latina
data_filtrada = data_2[data_2['Country Name'].isin(inter_lt)]
print(data_filtrada)
data_filtrada.columns

##Paises de america Latina en 2020
filt_2020 = data_filtrada[["Country Name","2020"]]
print(filt_2020)
pais = filt_2020['Country Name']
respais = pais.describe()
print(respais)

#**ANALISIS:
#Dado que la variable en estudio es el porcentaje del uso de internet, se obtuvo que el país con mayor porcentaje es Argentina al tener un 
#85,51% de uso lo cual es bastante, si lo comparamos con Ecuador, se observa que tiene un 70,70% de uso de internet.
#El análisis se lo realiza a 23 países de América Latina
#Por otro lado se tiene que el país con menor porcentaje de uso del internet es Haití con un 36,39%




#**************************************************PREGUNTAS***************************************

#   1RA  Pregunta

##¿Cuál es el valor promedio del indicador seleccionado entre los países de América Latina en el año 2020?

##Indicador:Porcentaje de uso de Internet.
mean_inter= filt_2020["2020"].mean()
#ANALISIS:
print("En promedio para el año 2020 en el porcentaje de uso de internet en america latina fue del ","%.2f"% mean_inter,"%")

variable = filt_2020['2020']
resumen = variable.describe()
print(resumen)
#**ANALISIS DEL AÑO 2020 PARA LATINOAMERICA:
#En general se tiene para el año 2020 que una vez cargados los datos del archivo Excel llamado INTER, 
#filtramos los países de América Latina y 
#seleccionamos los datos correspondientes al año 2020 y se realiza un análisis estadístico simple sobre 
#los nombres de los países seleccionados obteniendo los siguientes resultados importantes:
#Dado que la variable en estudio es el porcentaje del uso de internet, se obtuvo que el país con mayor porcentaje es Argentina al tener un 
#85,51% de uso lo cual es bastante, si lo comparamos con Ecuador, se observa que tiene un 70,70% de uso de internet.
#El análisis se lo realiza a 23 países de América Latina
#Por otro lado se tiene que el país con menor porcentaje de uso del internet es Haití con un 36,39% puede deberse a la varios
#factores como: infraestructura subdesarrollada, falta de recursos, entre otros



#  2DA Pregunta

##¿Cómo ha evolucionado este indicador a lo largo del tiempo en América Latina?

#Convertimos los datos a un formato mas manejable
data_melted = data_filtrada.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=['1960',
       '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
       '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978',
       '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
       '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
       '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
       '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
       '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'],
    var_name='Año'
)
print(data_melted)

#Calculamos el promedio para cada año de los paises de America latina y  lo insertamos en la base
data_melted['Promedio Regional%'] = data_melted.groupby('Año')['value'].transform('mean')
data_melted= data_melted.fillna(0)
data_melted
#Realizamos la grafica de la evolucion a lo largo del tiempo.
sns.lineplot(x='Año', y='Promedio Regional%', data=data_melted)
plt.title('Evolución del promedio regional')
plt.show()

#**ANALISIS:
#el gráfico muestra un crecimiento constante del promedio regional del uso de internet en América 
#Latina desde 2010 hasta 2024, por lo tanto ha experimentado un crecimiento significativo en los 
#últimos 14 años. El promedio regional del uso de internet ha aumentado de 30% a 80% en este período. 
#La brecha digital en América Latina y el Caribe es un problema importante, ya que existe una diferencia
# significativa en el acceso a internet entre los países de la región.




#  3RA Pregunta

##¿Cómo es el mapa de correlación entre los últimos 5 años de datos disponibles para los países de América Latina?
##data filtrada los ultimos 5 años
last5y = data_filtrada[["Country Name",'2018', '2019', '2020', '2021', '2022']]
last5y
last5y.dtypes 
# Calcula la matriz de correlación entre los ultimos 5 años
matriz_correlacion = last5y.select_dtypes(include=[np.number]).corr()
matriz_correlacion
# Creamos el mapa de calor 
sns.heatmap(matriz_correlacion, annot=True, cmap="YlGnBu")
plt.title('Mapa de Correlación')
plt.show()

#**ANALISIS:

#El mapa de correlación muestra que existe una fuerte correlación entre el uso de internet en 
#los países de América Latina. Los países con un mayor uso de internet tienden a tener una 
#correlación más fuerte con otros países con un alto uso de internet. Esto sugiere que los países 
#de la región están convergiendo en términos de su uso de internet. Los factores que pueden influir 
#en el uso de internet en los países de América Latina incluyen el PIB per cápita, el nivel de 
#educación, la infraestructura de telecomunicaciones y las políticas gubernamentales. La brecha 
#digital en América Latina y el Caribe es un problema importante, ya que existe una diferencia #
#significativa en el acceso a internet entre los países de la región