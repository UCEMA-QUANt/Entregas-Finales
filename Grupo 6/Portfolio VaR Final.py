"""
Cartera de CEDEARS
FECHA: 19/11/2020
"""
#pip install yahoofinance
#############################################################################
###############  DATOS FINANCIEROS Y SU PROCESAMIENTO  ######################
#############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns 
import scipy.stats as scs
import statsmodels.api as sm
import os
#import statsmodels.tsa.api as smt
import operator

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 82)
pd.set_option('precision', 3)


os.chdir("C:\\Users\\jzubillaga\\Desktop\\Var - Quant")

#IMPORTAMOS EL Excel
file = 'cartera.xlsx'
xls = pd.ExcelFile(file)
print(xls.sheet_names)
cartera = xls.parse('tenencias')
tickers = cartera["tickers"].tolist()
cartera = cartera.sort_values("tickers")
cartera = cartera.set_index("tickers")

FechaHoy = '2020-11-26'

#Importo Precios USD 252 ruedas
#cotizaciones = xls.parse('USD').head(252)
#cotizaciones = cotizaciones.set_index("Fecha")
#cotizacionhoy= cotizaciones.head(1) 

#Importo Precios USD Contado con Liquidar 252 ruedas
cotizaciones = xls.parse('USD CCL')
cotizaciones = cotizaciones.set_index("Fecha")
cotizacionhoy= cotizaciones.head(1) 

#Descargamos la data
#df_yahoo1 = yf.download(["LTC-USD"],start='2020-01-01',end='2020-08-19',progress=True)
df_yahoo2 = yf.download(tickers,start='2019-10-09',end= FechaHoy,auto_adjust=True,actions='inline',progress=True)
#df_yahoo3 = yf.download(['AAPL','MSFT','AMZN'],interval="1h",auto_adjust=True,actions='inline',progress=True,period="1mo")


#Actions="inline" me trae los dividendos y splits
#Si auto_adjust=True descargo precios ajustados.
#Si queremos descargar pago dedividendos and stock splits agregar actions='inline'.


#Me quedo solo con la Adj Close y le cambio el nombre a adj_close
precios = df_yahoo2
precios = precios.dropna()
precios = precios.loc[:, ['Close']]
precios= precios.xs('Close', axis=1) 
precios = precios.sort_values('Date')


#Paso los pesos a dólares
preciosdolarizados = pd.merge(left = precios, right=cotizaciones, how = 'left', left_on= precios.index, right_on='Fecha')
preciosdolarizados = preciosdolarizados.set_index("Fecha")
preciosdolarizados = preciosdolarizados.dropna()
preciosdolarizados ["MIRG.BA"] =  preciosdolarizados["MIRG.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["CEPU.BA"] =  preciosdolarizados["CEPU.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["PAMP.BA"] =  preciosdolarizados["PAMP.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["BYMA.BA"] =  preciosdolarizados["BYMA.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["GGAL.BA"] =  preciosdolarizados["GGAL.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["BPAT.BA"] =  preciosdolarizados["BPAT.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados ["AGRO.BA"] =  preciosdolarizados["AGRO.BA"]/preciosdolarizados["Precio Dólar"]
preciosdolarizados= preciosdolarizados.drop(['Precio Dólar'], axis=1)
preciosdolarizados= preciosdolarizados.tail(252)
precios= preciosdolarizados
precioshoy = precios.tail(1).T
precioshoy.columns = ["precio"]


#Defino dolares equivalentes a un millon de pésos
valorcartera = cartera ["Valor Cartera"]
valorcartera ["MIRG.BA"] =  valorcartera["MIRG.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["CEPU.BA"] =  valorcartera["CEPU.BA"]//cotizacionhoy ["Precio Dólar"]
valorcartera ["PAMP.BA"] =  valorcartera["PAMP.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["BYMA.BA"] =  valorcartera["BYMA.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["GGAL.BA"] =  valorcartera["GGAL.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["BPAT.BA"] =  valorcartera["BPAT.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["AGRO.BA"] =  valorcartera["AGRO.BA"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["WMT"] =  valorcartera["WMT"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["VIST"] =  valorcartera["VIST"]/cotizacionhoy ["Precio Dólar"]
valorcartera ["GOLD"] =  valorcartera["GOLD"]/cotizacionhoy ["Precio Dólar"]


carteraUSD = pd.concat([cartera,precioshoy],axis=1)
carteraUSD ["nominales"] = carteraUSD["Valor Cartera"] * carteraUSD["Porcentaje"] / carteraUSD["precio"]
carteraUSD["valorizado"]=carteraUSD["nominales"]*carteraUSD["precio"]  
VM=carteraUSD["valorizado"].sum()
VM

#Importo nominales a Cartera
cartera ["nominales"] = carteraUSD ["nominales"]

#Calculo los retornos logaritmos de cada activo

file = 'cartera.xlsx'
xls = pd.ExcelFile(file)
print(xls.sheet_names)
weights = xls.parse('tenencias')
weights = weights.sort_values("tickers")
weights= weights.drop(['conversion', 'Valor Cartera'], axis=1)
weights = weights.set_index("tickers")


retornos =  np.log(precios/precios.shift(1))
retornos = retornos.dropna()
retornos ["Portfolio Return"] = retornos.dot(weights)
retornos =  retornos.dropna()
#retornos = retornos.sort_values("Portfolio Return")
retornos.to_excel(r'C:\\Users\\jzubillaga\\Desktop\\Var - Quant"\\retornos.xlsx', index = True)


#############################################################################
#########################  Value At Risk  ###################################
#############################################################################
#VaR HISTÓRICO
HP = 1
significancia = 1

np.percentile(retornos ["Portfolio Return"], significancia)
VaR_Historico = VM * -(np.percentile(retornos ["Portfolio Return"], significancia) * np.sqrt(HP))

VaR_Historico


#VaR Parametrico
#Valor de la distr normal
z = scs.norm.ppf(0.99)
z
#VaR 1d
ds = retornos ["Portfolio Return"].std()
ds

VaR_Parametrico =  VM * (ds * z * np.sqrt(HP))
VaR_Parametrico

######## VaR EWMA ######
import math
import pandas as pd
import numpy as np
def CalculateEWMAVol (ReturnSeries, Lambda):   
    SampleSize = len(ReturnSeries)
    Average = ReturnSeries.mean()

    e = np.arange(SampleSize-1,-1,-1)
    r = np.repeat(Lambda,SampleSize)
    vecLambda = np.power(r,e)

    sxxewm = (np.power(ReturnSeries-Average,2)*vecLambda).sum()
    Vart = sxxewm/vecLambda.sum()
    EWMAVol = math.sqrt(Vart)

    return (EWMAVol)


def CalculateVol (R, Lambda):
    Vol = pd.Series(index=R.columns)
    for facId in R.columns:
        Vol[facId] = CalculateEWMAVol(R[facId], Lambda)

    return (Vol)


VaREWMA = VM * (z* CalculateEWMAVol(retornos ["Portfolio Return"], 0.94))#* np.sqrt(252/10)
CalculateEWMAVol(retornos["Portfolio Return"], 0.1) 
VaREWMA

#############################################################################
#########################  PORTAFOLIOS OPTIMOS  #############################
#############################################################################

#Importo librerias (e instalo alguna si hace falta)
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as scs

#lock o i loc para cortar por numero de registros

#Seteo los parametros
N_PORTFOLIOS = 10000 #cantidad de portafolios simulados
N_DAYS = 252
tickers = ['MIRG.BA', 'WMT', 'VIST', "GOLD", "CEPU.BA", "PAMP.BA", "BYMA.BA", "GGAL.BA", "BPAT.BA","AGRO.BA"]
START_DATE = '2010-10-01'
END_DATE = '2020-11-26'
n_assets = len(tickers)

#Descargamos la data
data = yf.download(tickers, start = START_DATE, end = END_DATE, progress=True)
data = data.dropna()
data = data.tail(252)
data = data.loc[:, ['Close']]
data = data.xs('Close', axis=1) #uso esto para borrar el doble titulo de cada columna
data.head()

datadolarizada = pd.merge(left = data, right=cotizaciones, how = 'left', left_on= data.index, right_on='Fecha')
datadolarizada = datadolarizada.set_index("Fecha")
datadolarizada ["MIRG.BA"] =  datadolarizada["MIRG.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["CEPU.BA"] =  datadolarizada["CEPU.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["PAMP.BA"] =  datadolarizada["PAMP.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["BYMA.BA"] =  datadolarizada["BYMA.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["GGAL.BA"] =  datadolarizada["GGAL.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["BPAT.BA"] =  datadolarizada["BPAT.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada ["AGRO.BA"] =  datadolarizada["AGRO.BA"]/datadolarizada ["Precio Dólar"]
datadolarizada= datadolarizada.drop(['Precio Dólar'], axis=1)
data = datadolarizada

data.to_excel(r'C:\Users\jzubillaga\Desktop\Var - Quant\data.xlsx', index = True)

#graficos la evolucion de los precios de los activos a considerar
data.plot(title='Precios de los activos considerados')

#muestro las n series temporales, pero normalizados para que empiecen en 100
(data / data.iloc[0] * 100).plot(figsize=(8, 6), grid=True,title='Precios de los activos considerados')

#calculo retornos logaritmicos

ret_l = np.log(data/data.shift(1))
ret_l = ret_l.dropna()
ret_l.head()

ret_l.plot(title='Retorno diario de los activos considerados')
ret_l.hist(bins=50, figsize=(9, 6))

#info estadistica adicional de cada activo
#defino funcion que luego voy a usar

#import numpy as np 
#from scipy.stats import kurtosis 
#from scipy.stats import skew 

def asim_y_curt(arr):  
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
      
for tck in tickers:
    print("\nResults for ticker %s" % tck)
    print(32 * "-")
    log_data = np.array(ret_l[tck].dropna())
    asim_y_curt(log_data)

#Retorno de cada activo
ret_prom = ret_l.mean()
ret_act1 = ret_prom[0] * N_DAYS
ret_act2 = ret_prom[1] * N_DAYS
ret_act3 = ret_prom[2] * N_DAYS
ret_act4 = ret_prom[3] * N_DAYS
ret_act5 = ret_prom[4] * N_DAYS
ret_act6 = ret_prom[5] * N_DAYS
ret_act7 = ret_prom[6] * N_DAYS
ret_act8 = ret_prom[7] * N_DAYS
ret_act9 = ret_prom[8] * N_DAYS
ret_act10 = ret_prom[9] * N_DAYS


#Matriz de varianzas y covarianzas
mat_cov = ret_l.cov()
desvios = ret_l.std()
desvio_act1 = desvios[0] * np.sqrt(N_DAYS)
desvio_act2 = desvios[1] * np.sqrt(N_DAYS)
desvio_act3 = desvios[2] * np.sqrt(N_DAYS)
desvio_act4 = desvios[3] * np.sqrt(N_DAYS)
desvio_act5 = desvios[4] * np.sqrt(N_DAYS)
desvio_act6 = desvios[5] * np.sqrt(N_DAYS)
desvio_act7 = desvios[6] * np.sqrt(N_DAYS)
desvio_act8 = desvios[7] * np.sqrt(N_DAYS)
desvio_act9 = desvios[8] * np.sqrt(N_DAYS)
desvio_act10 = desvios[9] * np.sqrt(N_DAYS)

#Primer intento al azar para construir un portfolio y encontrar su Sharpe Ratio
np.random.seed()
# Columna de Acciones
print('Acciones')
print(data.columns)
print('\n')
# Genero numeros random usando la distribución uniforme (0 a 1)
print('Numeros Aleatorios')
weights = np.array(np.random.random(n_assets))
print(weights)
print('\n')
# Transformo los numeros aleatorios a % (Cada numero / La suma de los n numeros)
print('Rebalanceo para que de 1')
weights = weights / np.sum(weights)
print(weights)
print('\n')
# Retorno esperado anual del portafolio
print('Retorno esperado del Portfolio')
exp_ret = np.sum(ret_prom * weights) * N_DAYS
print(exp_ret)
print('\n')
# Volatilidad esperada anual
print('Volatilidad Esperada')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(mat_cov * N_DAYS, weights)))
print(exp_vol)
print('\n')
# Sharpe Ratio
SR = exp_ret/exp_vol
print('Sharpe Ratio')
print(SR)

#Comenzamos las simulaciones de portafolios inicializando las matrices
all_weights = np.zeros((N_PORTFOLIOS,len(data.columns)))
ret_arr = np.zeros(N_PORTFOLIOS)
vol_arr = np.zeros(N_PORTFOLIOS)
sharpe_arr = np.zeros(N_PORTFOLIOS)
#seq = list(range(0,N_PORTFOLIOS)) #lista que arranca en 0 y termina en cant de port 

#Por cada portafolio de las N simulaciones, calculamos las ponderaciones, 
#el retorno, la volatilidad y el SR.
for i in range(N_PORTFOLIOS):
    weights=np.array(np.random.random(n_assets))
    weights=weights/np.sum(weights)
    all_weights[i,:] = weights
    ret_arr[i] = np.sum((ret_prom * weights) * N_DAYS)
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(mat_cov * N_DAYS, weights)))
    
for i in range(N_PORTFOLIOS):
    sharpe_arr[i] =  ret_arr[i]/vol_arr[i]
    
#Analicemos los resultados
print("El Sharpe Ratio Max es: " + str(sharpe_arr.max()))
print("en el Portfolio número: " + str(sharpe_arr.argmax()))
print("La Menor Volatilidad es: " + str(vol_arr.min()))
print("en el Portfolio número: " + str(vol_arr.argmax()))
print()
print('Proporciones:\n')
print("AGRO.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][0]*100,2)) + "%")
print("BPAT.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][1]*100,2)) + "%")
print("BYMA.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][2]*100,2)) + "%")
print("CEPU.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][3]*100,2)) + "%\n")
print("GGAL.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][4]*100,2)) + "%\n")
print("GOLD: " + str(round(all_weights[sharpe_arr.argmax(),:][5]*100,2)) + "%\n")
print("MIRG.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][6]*100,2)) + "%\n")
print("PAMP.BA: " + str(round(all_weights[sharpe_arr.argmax(),:][7]*100,2)) + "%\n")
print("VIST: " + str(round(all_weights[sharpe_arr.argmax(),:][8]*100,2)) + "%\n")
print("WMT: " + str(round(all_weights[sharpe_arr.argmax(),:][9]*100,2)) + "%\n")

#Graficamente
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]
min_vol = vol_arr[vol_arr.argmin()]
ret_min_vol = ret_arr[vol_arr.argmin()]

plt.figure(figsize=(20,10))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='RdYlGn')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatilidad', fontsize=25, color='black')
plt.ylabel('Retorno', fontsize=25, color='black')
plt.title('Portafolios simulados: ',fontsize=25, color='black')
plt.scatter(max_sr_vol,max_sr_ret,c='red',marker='*',s=250,label='Max Sharpe Ratio')
plt.scatter(min_vol,ret_min_vol,c='black',marker='X',s=250,label='Min Vol')
plt.scatter(desvio_act1,ret_act1,c='black',marker='v',s=200,label='AGRO.BA')
plt.scatter(desvio_act2,ret_act2,c='black',marker='>',s=200,label='BPAT.BA')
plt.scatter(desvio_act3,ret_act3,c='black',marker='d',s=200,label='BYMA.BA')
plt.scatter(desvio_act4,ret_act4,c='black',marker='D',s=200,label='CEPU.BA')
plt.scatter(desvio_act5,ret_act5,c='black',marker='+',s=200,label='GGAL.BA')
plt.scatter(desvio_act6,ret_act6,c='black',marker='_',s=200,label='GOLD')
plt.scatter(desvio_act7,ret_act7,c='black',marker='o',s=200,label='MIRG.BA')
plt.scatter(desvio_act8,ret_act8,c='black',marker='p',s=200,label='PAMP.BA')
plt.scatter(desvio_act9,ret_act9,c='black',marker='<',s=200,label='VIST')
plt.scatter(desvio_act10,ret_act10,c='black',marker='P',s=200,label='WMT')
plt.tick_params(axis='both', colors='blue', labelsize=20)
plt.legend(loc='center right', fontsize=17)
plt.show()

#Resolución matemática
#Función que reciba las proporciones del portafolio y devuelva el retorno, 
#la volatilidad y el SR
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(ret_l.mean() * weights) * N_DAYS
    vol = np.sqrt(np.dot(weights.T, np.dot(ret_l.cov() * N_DAYS, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

#Función a minimizar, Sharpe Ratio Negativo
def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1

#Restricción para minimizar
def check_sum(weights):
    return np.sum(weights) - 1

cons = ({'type':'eq','fun': check_sum})

#Las proporciones deben estar entre 0 y 1, no permito ir short.
bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),(0, 1),(0, 1),(0, 1),(0, 1))

#Optimizamos el portafolio
init_guess = [0.05, 0.10, 0.10,0.05, 0.05, 0.20, 0.3, 0.01, 0.02, 0.13]
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

#Resultados
print('Portafolio Optimo (Periodo Mayo 2010 - Noviembre 2020):\n')
print('Proporciones:\n')
print("AGRO.BA: " + str(round(opt_results.x[0]*100,2)) + "%")
print("BPAT.BA: " + str(round(opt_results.x[1]*100,2)) + "%")
print("BYMA.BA: " + str(round(opt_results.x[2]*100,2)) + "%")
print("CEPU.BA: " + str(round(opt_results.x[3]*100,2)) + "%\n")
print("GGAL.BA: " + str(round(opt_results.x[4]*100,2)) + "%\n")
print("GOLD: " + str(round(opt_results.x[5]*100,2)) + "%\n")
print("MIRG.BA: " + str(round(opt_results.x[6]*100,2)) + "%\n")
print("PAMP.BA: " + str(round(opt_results.x[7]*100,2)) + "%\n")
print("VIST: " + str(round(opt_results.x[8]*100,2)) + "%\n")
print("WMT: " + str(round(opt_results.x[9]*100,2)) + "%\n")
print('Metricas:\n')
print("RETORNO MEDIO: " + str(round(get_ret_vol_sr(opt_results.x)[0]*100,2)) + "%")
print("VOLATILIDAD: " + str(round(get_ret_vol_sr(opt_results.x)[1]*100,2)) + "%")
print("SHARPE: " + str(get_ret_vol_sr(opt_results.x)[2]))

