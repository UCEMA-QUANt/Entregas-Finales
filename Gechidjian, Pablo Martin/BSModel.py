import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from math import sqrt, exp
from mpl_toolkits.mplot3d import Axes3D

from math import factorial

class BS:
    '''
    Calculate the option price according to the Black-Scholes-Merton model.

    Attributions
    ============
    call: True if the option is call, False otherwise (boolean)
    stock: Price of the underlying security (float)
    strike: Strike price of the option (float)
    maturity: Time to maturity of the option in years (float)
    interest: Annual interest rate expressed as decimal (float)
    volatility: Annual volatility expressed as decimal (float)
    dividend: Annual dividend yield expressed as decimal (float)

    Methods
    =======
    price: Returns the price of the option according to the Black-Scholes-Merton model.
    '''


    def __init__(self, type, stock, strike, maturity, interest, volatility, dividend):
        self.type = type

        self.stock = stock
        self.strike = strike
        self.maturity = maturity
        self.interest = interest    #rfrate
        self.volatility = volatility
        self.dividend = dividend    # no usado
        self.d1 = (self.volatility * sqrt(self.maturity)) ** (-1) * (np.log(self.stock / self.strike) + (
                    self.interest - self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * sqrt(self.maturity)

    def option_value(self):
        if self.type=="call":
            return exp(-self.dividend * self.maturity) * norm.cdf(self.d1) * self.stock - norm.cdf(
                self.d2) * self.strike * exp(-self.interest * self.maturity)

        elif self.type =="put":
            return norm.cdf(-self.d2) * self.strike * exp(-self.interest * self.maturity) - \
               norm.cdf(-self.d1) * self.stock * exp(-self.dividend * self.maturity)
        else: print("Por favor, ingrese unicamente ingrese call o put en tipo de opcion")

class JDM(BS):
    def __init__(self, type, stock, strike, maturity, interest, volatility, dividend, Lambda, beta, D):
        self.type = type
        self.S = stock
        self.K = strike
        self.TTM = maturity
        self.r = interest
        self.volatility = volatility
        '''
        self.d1 = (self.volatility * sqrt(self.maturity)) ** (-1) * (np.log(self.stock / self.strike) + (
                self.interest - self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * sqrt(self.maturity)
        '''
        self.jump_rate = Lambda
        self.jump_intensity_mean = 1+ beta
        self.jump_intensity_sd = D

    def price(self,Nterms):
        alfa = 2 * np.log(self.jump_intensity_mean) - 0.5 * np.log(
            self.jump_intensity_sd ** 2 + self.jump_intensity_mean ** 2)
        tita2 = -2 * np.log(self.jump_intensity_mean) + np.log(
            self.jump_intensity_sd ** 2 + self.jump_intensity_mean ** 2)

        price = 0
        for n in range(0,Nterms+1):
            #d1n = (np.log(self.stock/self.strike) + (self.r + 0.5*self.volatility**2 - self.jump_intensity_mean*self.jump_rate)*self.TTM + n*(alfa+tita2)) / (np.sqrt(self.volatility**2*self.TTM+n*tita2))
            #term_n = ((self.jump_rate*self.TTM)/factorial(n)) * np.exp(-(self.jump_rate*self.TTM)) * (self.stock*np.exp(-self.jump_intensity_mean*self.jump_rate*self.TTM+n*alfa+0.5*n*tita2)*norm.cdf(d1n))
            Lambda_p = self.jump_rate*(self.jump_intensity_mean)
            r_n = self.r + (n/self.TTM) * (alfa + tita2/2) - self.jump_rate*(self.jump_intensity_mean-1)
            sigma_n = np.sqrt(self.volatility**2 + n*tita2/self.TTM)
            # type, stock, strike, maturity, interest, volatility, dividend
            term_n = (((Lambda_p*self.TTM)**n) /factorial(n))* np.exp(-(Lambda_p*self.TTM))    * BS("call", self.S, self.K, self.TTM, r_n, sigma_n, 0 ).option_value()
            price += term_n
        return price




if __name__=="__main__":
    ' type, stock, strike, maturity, interest, volatility, dividend)'
    my_bs_model = BS("call",100,90,1,0.05,0.1,0)
    print(f"El {my_bs_model.type} de strike $ {my_bs_model.strike} y maturity a {round(my_bs_model.maturity,4)*252} dias vale: $ {round(my_bs_model.option_value(),2)}" )

    my_bs2 = BS("call", 100, 90, 1, 0.05, 0.4, 0)
    print("$ "+str(my_bs2.option_value()))

    # type, stock, strike, maturity, interest, volatility, dividend, Lambda, beta, D)
    my_jdm_model = JDM("call", 100, 90, 1, 0.1, 0.1, 0 , 1, 1, 0.2)
    precio_jdm = my_jdm_model.price(100)
    print(f"El {my_jdm_model.type} de strike $ {my_jdm_model.K} y maturity a {round(my_jdm_model.TTM, 4) * 252} dias vale: $ {round(precio_jdm, 2)}")


    '''
    ### SUPERFICIES DE SENSIBILIDAD ###
    # Modelo de Black Scholes
    from BSModel import BS

    # Create arrays with the different input values for each variable
    S = np.linspace(45, 135, 50)  # stock price
    T = np.linspace(0.01, 3, 50)  # time to maturity
    s = np.linspace(0.001, 0.8, 50)  # volatility

    # TTM
    # Calculate call price for different stock prices and time to maturity
    ct = np.array([])
    for i in range(0, len(T)):
        # type, stock, strike, maturity, interest, volatility, dividend
        ct = np.append(ct, BS("call", S, 90, T[i], 0.05, 0.3, 0.02).price(), axis=0)
    ct = ct.reshape(len(S), len(T))

    # Generate 3D graph for time to maturity
    X1, Y1 = np.meshgrid(S, T)

    figct = plt.figure()
    ax = Axes3D(figct)
    ax.plot_surface(X1, Y1, ct, rstride=1, cstride=1, cmap=cm.coolwarm, shade='interp')
    ax.view_init(27, -125)
    plt.title('Call Option Price wrt Time to Maturity')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Call price')

    plt.show()

    # Volatilidad
    # Calculate call price for different stock prices and volatility
    cs = np.array([])
    for i in range(0, len(s)):
        # # type, stock, strike, maturity, interest, volatility, dividend
        cs = np.append(cs, BS("call", S, 90, 1, 0.05, s[i], 0).price(), axis=0)
    cs = cs.reshape(len(S), len(s))

    # Generate 3D graph for volatility
    X2, Y2 = np.meshgrid(S, s)

    figcs = plt.figure()
    ax = Axes3D(figcs)
    ax.plot_surface(X2, Y2, cs, rstride=1, cstride=1, cmap=cm.coolwarm, shade='interp')
    ax.view_init(27, -125)
    plt.title('Call Option Price wrt Volatility')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Call price')

    plt.show()
    '''