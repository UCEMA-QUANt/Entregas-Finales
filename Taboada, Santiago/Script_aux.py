# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:09:50 2020

@author: santi
"""

import datetime as dt
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
import math
import scipy.optimize as optimize
from scipy.optimize import curve_fit

####Interpolacion####

#Interpolacion Poligonal

def InterpoladorPoligonal(x,y):
    f=interpolate.interp1d(x,y)
    xForPlot=np.linspace(0.1,30,1001,endpoint=True)
    plt.figure(1)
    plt.plot(x,y,"o",label="data")
    plt.plot(xForPlot,f(xForPlot),"-",label="poligonal")
    plt.legend(loc="upper left")


#Interpolacion Previo

def InterpoladorEscalonPrevio(x,y):
    f=interpolate.interp1d(x,y,kind="previous")
    xForPlot=np.linspace(0.1,30,1001,endpoint=True)
    plt.figure(1)
    plt.plot(x,y,"o",label="data")
    plt.plot(xForPlot,f(xForPlot),"-",label="escalon previous")
    plt.legend(loc="upper left")


#Interpolacion Posterior

def InterpoladorEscalonPosterior(x,y):
    f=interpolate.interp1d(x,y,kind="next")
    xForPlot=np.linspace(0.1,30,1001,endpoint=True)
    plt.figure(2)
    plt.plot(x,y,"o",label="data")
    plt.plot(xForPlot,f(xForPlot),"-",label="escalon next")
    plt.legend(loc="upper left")


#Splines

def Interpl(x,y):
    plt.figure(figsize=(9,5))
    tck1 = interpolate.splrep(x, y, s=0)
    tck2 = interpolate.splrep(x, y, s=1)
    xForPlot = np.linspace(0, 30, num=1001, endpoint=True)
    yForPlot1 = interpolate.splev(xForPlot, tck1, der=0)
    yForPlot2 = interpolate.splev(xForPlot, tck2, der=0)
    plt.plot(x, y, 'o', linewidth=0,label="Datos al 15 de septiembre de 2017")
    plt.plot(x, y, '-',color="b", label="Curva original")
    plt.plot(xForPlot, yForPlot2, '--', color="r", label="Spline smoothness=1")
    plt.title("Curva del Tesoro interpolada vía splines")
    plt.xlabel("Maturity (en años)")
    plt.ylabel("Tasa de interes (en %)")
    plt.legend()
    
####Nelson-Siegel#####

def nelsonSiegelEnLaForward(T, b0, b1, b2, t1):
    return b0 + (b1 + b2 * T) * np.exp(-T/t1)

def nelsonSiegelEnElRate(T, b0, b1, b2, t1):
    return b0 + (b1 + b2) * t1/T * (1 - np.exp(-T/t1)) - b2 * np.exp(-T/t1)

def plotNelsonSiegel():
    b0 = 7
    b1 = -4.13
    T = np.linspace(0, 20, num=1001, endpoint=True)
    plt.figure(figsize=(9,5))
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=1.5, t1=2), color='black', label='beta2=1.5')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=0.5, t1=2), color='cyan', label='beta2=0.5')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=0.0, t1=2), color='red', label='beta2=0.0')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=-0.5, t1=2), color='blue', label='beta2=-0.5')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=-2.0, t1=2), color='green', label='beta2=-2.0')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=-3.0, t1=2), color='orange', label='beta2=-3.0')
    plt.plot(T, nelsonSiegelEnLaForward(T, b0, b1, b2=-4.0, t1=2), color='gray', label='beta2=-4.0')
    plt.title("Curva del Tesoro vía Nelson-Siegel c/ diferentes betas")
    plt.xlabel("Maturity (en años)")
    plt.ylabel("Tasa de interes (en %)")
    plt.legend()
    plt.show()
    
def fitNelsonSiegel(x,y):
    xdata = x
    ydata = y
    popt, pcov = curve_fit(nelsonSiegelEnElRate, xdata, ydata)
    [b0, b1, b2, t1] = popt
    print([b0, b1, b2, t1])
    
    T = np.linspace(0, 40, num=1001, endpoint=True)
    plt.figure(figsize=(9,5))
    plt.plot(xdata, ydata, color='black', marker="o", linewidth=0,label='Al 10 de agosto de 2017')
    plt.plot(T, nelsonSiegelEnElRate(T, b0, b1, b2, t1), color='r', label='Nelson-Siegel Fit')
    plt.title("Curva del Tesoro ajustada vía Nelson-Siegel")
    plt.xlabel("Maturity (en años)")
    plt.ylabel("Tasa de interes (en %)")
    plt.xlim(0,35)    
    plt.legend()

####Modelo Cox Ingersoll Ross####

def cir_model(n_years=10,n_scenarios=1,a=0.2,b=1.5,sigma=0.05,steps_per_year=12,r_0=None):
    #Un seguro por si no le meto de input la tasa incial
    if r_0 is None: r_0 = b
    r_0 = ann_to_inst(r_0)
    #Defino el valor del diferencial del tiempo (dt)
    dt=1/steps_per_year
    #Defino el numero de pasos por año    
    num_steps=int(n_years*steps_per_year)+1 #n_years puede ser una variable float
    
    
    #Defino la parte estocastica de la ecuacion
    rw=np.random.normal(0,1,size=(num_steps,n_scenarios))
    #Creo una matriz vacia para ir rellenandola; con la forma de la matriz rw
    rates=np.empty_like(rw)
    #Meto la tasa inicial en la primera posicion en la matriz rates
    rates[0]=r_0
    
    #Genero los precios de los ZCB
    
    gamma=math.sqrt(a**2+sigma**2)
    precios=np.empty_like(rw)
    
    def precio(ttm,r):
        _A=(2*gamma*math.exp((gamma+a)*ttm/2)/((gamma+a)*math.expm1(gamma*ttm)+2*gamma))**(2*a*b/(sigma**2))
        _B=(2*math.expm1(gamma*ttm))/((gamma+a)*math.expm1(gamma*ttm)+2*gamma)
        _P=_A*(np.exp(-_B*r))
        #_R=(-np.log(_A)+(_B*r))/ttm
        return _P
    precios[0]=precio(n_years,r_0) 

    
    #Lleno los resultados de la ecuacion CIR en la matriz rates
    for i in range(1,num_steps):
        r_t=rates[i-1]
        dr=a*(b-r_t)*dt+sigma*np.sqrt(r_t)*rw[i]
        rates[i]=abs(r_t+dr)
        #genero los precios al momento t
        precios[i]=precio(n_years-i*dt,rates[i])
    
    #data=inst_to_ann(rates)
    #Tasas
    df_output=pd.DataFrame(data=inst_to_ann(rates),index=range(num_steps),columns=range(1,n_scenarios+1))
    #Precios ZCB
    precios=pd.DataFrame(data=precios,index=range(num_steps),columns=range(1,n_scenarios+1))    
            
    return df_output,precios

####Curvas de ZCB####

def curva_zcb(ttm,r,sigma,a,b):
    
    if isinstance(r,pd.Series) and isinstance(ttm,list):
        
        gamma=math.sqrt((math.pow(a,2)+2*math.pow(sigma,2)))      
        #rates_avg=r.transpose().mean()
        precios_zcb=np.empty(shape=(len(r),len(ttm)))
        rates_zcb=np.empty(shape=(len(r),len(ttm)))
        
        for i in range(0,len(r)):
            for t in range(0,len(ttm)):
                _A=(2*gamma*math.exp((gamma+a)*ttm[t]/2)/((gamma+a)*math.expm1(gamma*ttm[t])+2*gamma))**(2*a*b/(sigma**2))
                _B=(2*math.expm1(gamma*ttm[t]))/((gamma+a)*math.expm1(gamma*ttm[t])+2*gamma)
                precios_zcb[i][t]=_A*(np.exp(-_B*r.loc[i]))
                rates_zcb[i][t]=(-np.log(_A)+(_B*r.loc[i]))/ttm[t]
        
        precios_zcb=pd.DataFrame(data=precios_zcb,index=r.index)
        rates_zcb=pd.DataFrame(data=rates_zcb,index=r.index)
        
        return rates_zcb
    
    else:
        gamma=math.sqrt((math.pow(a,2)+2*math.pow(sigma,2)))
        _A=(2*gamma*math.exp((gamma+a)*ttm/2)/((gamma+a)*math.expm1(gamma*ttm)+2*gamma))**(2*a*b/(sigma**2))
        _B=(2*math.expm1(gamma*ttm))/((gamma+a)*math.expm1(gamma*ttm)+2*gamma)
        #_P=_A*(np.exp(-_B*r))
        _R=(-np.log(_A)+(_B*r))/ttm
        
        return _R
    
#####Formulas de calculo de bonos#######


def inst_to_ann(r):
    """
    Convierto la tasa instantanea en tasa anual
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    Convert la tasa anual en una tasa instantanea
    """
    return np.log1p(r)


def cashflow_bono(maturity,principal=100,cupon=2,cupon_per_year=2):
    """
    Calcula los cashflows de un bono
    """   
    #n_cupon=round(maturity*cupon_per_year)
    n_cupon=maturity*cupon_per_year
    if n_cupon==0: n_cupon==1
    cupon_amort=principal*(cupon/100)/cupon_per_year
    cupon_times=np.arange(1,n_cupon+1)
    cash_flows=pd.Series(data=cupon_amort,index=cupon_times)
    cash_flows.iloc[-1]+=principal
    
    #Devuelve un objeto Pandas Series
    
    return cash_flows


def descuento(t,r,strip=0):
    """
    Calcula el precio de un bono a descuento que paga 1 unidad monetaria ($) en el periodo t
    """
    #Paga a la tasa r, que es la tasa por periodo
    if strip==0:

        descuentos = pd.DataFrame([(1+(r/100))**-i for i in t])
        descuentos.index=t
        
        #Devuelve un objeto Pandas DataFrame
        
        return descuentos
    
    else: 
        descuentos = pd.DataFrame((1/(1+(r/100))))
        #descuentos.index=t
        
        return descuentos                                   
     
                              
def pv(flows, r,strip=0):
    """
    Calcula el valor presente de una secuencia de cahsflows 
    """
    if strip==0:
    
        dates = flows.index
        discounts = descuento(dates, r)
        
        return discounts.multiply(flows, axis='rows').sum()
   
    else: 
        discounts=descuento(flows,r,strip=1)
        
        return discounts.multiply(flows,axis="rows")


def bon_precio(maturity,principal=100,cupon=2,cupon_per_year=2,tasa_desc=1.5):
    """
    Calcula el precio del bono descontando los pagos 
    """ 
    cash_flow=cashflow_bono(maturity,principal,cupon,cupon_per_year)
    
    
    #Devuelve un objeto Pandas Series con los cashflows descontados

    return pv(cash_flow,tasa_desc/cupon_per_year)

    
def bon_precio_df(maturity,principal=100,cupon=2,cupon_per_year=2,tasa_desc=1.5):
    """""
    Calcula el precio del bono descontando los pagos de manera vectorizada
    """
    if isinstance(tasa_desc,pd.DataFrame):
        pricing_dates=tasa_desc.index
        precios=pd.DataFrame(index=pricing_dates,columns=tasa_desc.columns)
        for t in pricing_dates:
            try:
                precios.loc[t] = bon_precio(maturity-t/12,principal,cupon,cupon_per_year,
                                        tasa_desc.loc[t]*100)
            except: precios.loc[t]=principal
        return precios     
    else: # caso base --> Valuacion de un periodo
        if maturity <= 0: return principal
        cash_flow = cashflow_bono(maturity,principal,cupon,cupon_per_year)  
        return pv(cash_flow, tasa_desc/cupon_per_year)

    
def bon_ytm(precio,maturity,cup,cupon_per_year,principal=100,guess=0.05):
    """
    Calcula la YTM o TIR de un bono
    """
       
    freq=float(cupon_per_year)
    precio=float(precio)
    periods=maturity*freq
    cupon_value=cup/100.*principal
    dt=[(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: \
    sum([cupon_value/freq/(1+y/freq)**(freq*t) for t in dt]) +\
    principal/(1+y/freq)**(freq*maturity) - precio
    
    return optimize.newton(ytm_func,guess)
    
    
def macaulay_duration(cashflow, tasa_desc):
    """
    Calcula la Macaulay Duration de una serie de cash flows
    """
    cash_flow_descontado=descuento(cashflow.index,tasa_desc,strip=0)*pd.DataFrame(data=cashflow)
    ponderacion=cash_flow_descontado/cash_flow_descontado.sum()
   
    return np.average(cashflow.index,weights=ponderacion.iloc[:,0])


def bon_convexity(precio, maturity, cupon, cupon_per_year,principal=100, dy=0.01):
    """
    Calcula la Convexity de un bono
    """
    
    ytm = bon_ytm(precio,maturity, cupon, cupon_per_year,principal)

    ytm_minus = ytm - dy    
    price_minus = bon_precio(maturity, principal, cupon, cupon_per_year,ytm_minus*100)
    
    ytm_plus = ytm + dy
    price_plus = bon_precio(maturity, principal, cupon, cupon_per_year,ytm_plus*100)
    
    convexity = (price_minus + price_plus - 2*precio)/(precio*dy**2)
    return convexity


def bon_mod_duration(precio, maturity, cupon, cupon_per_year,principal=100,dy=0.01):
    ytm = bon_ytm(precio, maturity, cupon, cupon_per_year,principal)
    
    ytm_minus = ytm - dy    
    price_minus = bon_precio(maturity, principal, cupon, cupon_per_year,ytm_minus*100)
    
    ytm_plus = ytm + dy
    price_plus = bon_precio(maturity, principal, cupon, cupon_per_year,ytm_plus*100)
    
    mduration = (price_minus-price_plus)/(2*precio*dy)
    return mduration


def pagos_frac(cupones,_date,daycount=360):
   
    pagos_frac=[]
    for i in range(0,len(cupones)):
         delta=cupones[i]-_date
         delta_frac=delta.days/daycount
         pagos_frac.append(delta_frac)
    cupon_times=np.arange(1,len(cupones)+1)
    pagos_f=pd.Series(data=pagos_frac,index=cupon_times)
    pagos_f[pagos_f<0]=0
    return pagos_f