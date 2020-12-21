import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import norm
from scipy import interpolate
from math import sqrt, exp
from mpl_toolkits.mplot3d import Axes3D


class JumpDiffusionProcess():
    '''Jump Diffusion Process in Risk Neutral Measure.'''

    def __init__(self,Nsteps=252,Nsim=100,process_params={"S0":100,"r":0.05,"volatility":0.3,"lamb":0.25,"beta":0.1,"D":0.1,"TTM":1},seed=None):
        self.initial_value = process_params["S0"]         # S_0

        self.rfrate = process_params["r"]                 #r
        self.volatility = process_params["volatility"]             # sigma
        self.jump_rate = process_params["lamb"]              # lambda
        self.jump_intensity_mean = process_params["beta"]  # beta: intensidad media del salto
        self.jump_intensity_std = process_params["D"]     # D: desvio de intensidad del salto
        self.horizon = process_params["TTM"]                # T

        self.Nsteps = Nsteps    # Cant. pasos entre comienzo(=0) y self.horizon
        self.Nsim = Nsim        # Cant. trayectorias de Nsteps pasos

        self.simulated_paths = self.sim_path()
        self.final_value = self.final_value()   # S_T

        self.seed = seed

    def sim_path(self):
        '''Devuelve una muestra de Nsim valores del estado final del proceso, alcanzado en Nsteps pasos'''

        # TamaNio de 1 paso
        dt = self.horizon / self.Nsteps

        '''Parametros de la Normal(alfa,tita2) asociada a la Lognormal(1+beta,D) de intensidad de saltos'''
        alfa = 2*np.log(1+self.jump_intensity_mean)-0.5*np.log(self.jump_intensity_std**2+(1+self.jump_intensity_mean)**2)
        tita2 = -2*np.log(1+self.jump_intensity_mean)+np.log(self.jump_intensity_std**2+(1+self.jump_intensity_mean)**2)

        '''
        Genero un array de ceros de Nsim x (Nsteps+1) para ir guardando el 
        valor de cada simulacion del proceso (columnas) en cada instante (filas).
        '''
        simulated_paths = np.zeros([self.Nsim, self.Nsteps + 1])

        # Reemplazar la primer columna del array con la condicion inicial de S_0 en todas las Nsim simulaciones
        simulated_paths[:, 0] = self.initial_value

        # Genero los 3 vectores aleatorios asociados cada uno a su factor de riesgo
        R_1 = np.random.normal(size=[self.Nsim, self.Nsteps])                         # Fuente de aleatoriedad asociado al movimiento browniano W(t)
        R_2 = np.random.normal(size=[self.Nsim, self.Nsteps])                         # Fuente de aleatoriedad asociada a la log-intensidad del salto Z=ln(U)=ln(1+Y)
        NPoisson = np.random.poisson(self.jump_rate * dt, [self.Nsim, self.Nsteps])   # Fuente de aleatoriedad asociado al arribo de saltos N(t)

        # Simulacion del proceso y carga de la grilla con Nsim caminos de Nsteps pasos
        for i in range(self.Nsteps):
            simulated_paths[:, i + 1] = \
                simulated_paths[:, i] * np.exp(
                (self.rfrate - (self.volatility ** 2) / 2 - self.jump_intensity_mean * self.jump_rate) * dt
                + self.volatility * np.sqrt(dt) * R_1[:, i]
                + alfa * NPoisson[:, i]
                + np.sqrt(tita2) * np.sqrt(NPoisson[:, i]) * R_2[:, i]
            )
        return simulated_paths

    def final_value(self):
        # Me quedo con los precios de S en maturity t=T
        final_value = self.simulated_paths[:, -1]
        return final_value

    def show_process(self):
        ############ CONFIGURACION DEL GRAFICO ############
        # Paleta de colores, tamaNio de la ventana, defino ejes
        sns.set(palette='viridis')
        plt.figure(figsize=(10, 8))
        ax = plt.axes()

        # Generacion del eje temporal
        t = np.linspace(0, self.horizon, self.Nsteps + 1) * self.Nsteps

        # Grafico de las trayectorias simuladas
        jump_diffusion = ax.plot(t, self.simulated_paths.transpose());

        # Afinar grosor del grafico de las trayectorias
        plt.setp(jump_diffusion, linewidth=1);

        # Titulo y leyenda de ejes x e y
        '''
        ax.set(title=
               "Simulacion de Monte Carlo de trayectorias del modelo de saltos de equity de Merton"
                f"\n$S_0$ = {self.initial_value}, $\mu$ = $r$ = {self.rfrate}, $\sigma$ = {self.volatility}, "
               +r"$\beta$ "+f"= {self.jump_intensity_mean}, D = {self.jump_intensity_std}, "
                f"$\lambda$ = {self.jump_rate}, T = {self.horizon}, Nsteps = {self.Nsteps}, Nsim = {self.Nsim}" , \
               xlabel='Tiempo [dias]', ylabel='Precio del activo subyacente')
        '''
        title = "Simulacion de Monte Carlo de trayectorias del modelo de saltos de equity de Merton"\
                f"\n$S_0$ = {self.initial_value}, TTM = {self.horizon}, $\mu$ = $r$ = {self.rfrate}, " \
                f"$\sigma$ = {self.volatility}, $\lambda$ = {self.jump_rate}, "+rf"$\beta$ = {self.jump_intensity_mean}, "\
                f"D = {self.jump_intensity_std}, Steps = {self.Nsteps}, Sim. = {self.Nsim}"

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Tiempo [dias]', fontsize=20)
        plt.ylabel('Precio del activo subyacente', fontsize=20)
        plt.title(title, fontsize=20, fontweight="bold")
        #plt.legend(fontsize=20)

        # Mostrar grafico
        plt.show()

    def final_histogram(self,bins=20):
        # Curiosidad: Histograma de precios finales S_T - (Apagado)
        plt.hist(self.final_value, bins)
        plt.show()

    def theoretical_return_mean(self,t_steps):
        '''
        Devuelve la media teorica del RENDIMIENTO del proceso en el instante t.
        Verificar las unidades del ancho de los pasos antes de usar
            Por ejemplo, si T = 1 y Nsteps = 252, cada dt sera de 1 dia

        E[ ln(S(t)/S0) ] = (r - (sigma**2)/2 - lambda * beta) * t + alfa * lambda * t
        '''
        '''Parametros de la Normal(alfa,tita2) asociada a la Lognormal(1+beta,D) de intensidad de saltos'''
        alfa = 2 * np.log(1 + self.jump_intensity_mean) - \
               0.5 * np.log(self.jump_intensity_std ** 2 + (1 + self.jump_intensity_mean) ** 2)
        tita2 = -2 * np.log(1 + self.jump_intensity_mean) + \
                np.log(self.jump_intensity_std ** 2 + (1 + self.jump_intensity_mean) ** 2)
        return (self.rfrate - (self.volatility ** 2) / 2 - self.jump_intensity_mean * self.jump_rate) * t_steps + alfa*self.jump_rate*t_steps

    def theoretical_return_variance(self,t_steps):
        '''
        Devuelve la varianza teorica del RENDIMIENTO del proceso a en el instante t.
        Verificar las unidades del ancho de los pasos antes de usar
        Por ejemplo, si T = 1 y Nsteps = 252, cada dt sera de 1 dia

        Var[ ln(S(t)/S0) ] = (sigma^2 + lambda * (alfa^2 + tita^2)) * t
        '''
        alfa = 2 * np.log(1 + self.jump_intensity_mean) - 0.5 * np.log(
            self.jump_intensity_std ** 2 + (1 + self.jump_intensity_mean) ** 2)
        tita2 = -2 * np.log(1 + self.jump_intensity_mean) + np.log(
            self.jump_intensity_std ** 2 + (1 + self.jump_intensity_mean) ** 2)
        return (self.volatility**2 + self.jump_rate * (alfa**2 + tita2))*t_steps #tita2 ya es tita^2





class Option:
    def __init__(self,stochastic_process,option_params={"type":"call","strike":90,"TTM":1}):
        self.stochastic_process = stochastic_process    # Dinamica del activo subyacente
        self.type = option_params["type"]  # call o put
        self.K = option_params["strike"]      # Strike price
        self.TTM = option_params["TTM"]    # Time to maturity

    def option_value(self):
        '''Devuelve el valor de una opcion Europea de strike X, maturity T, y vector de precios posibles S_T'''
        if self.type == "call":
            payoff = np.maximum(self.stochastic_process.final_value - self.K, 0)
        elif self.type == "put":
            payoff = np.maximum(self.K - self.stochastic_process.final_value, 0)
        else:
            return "Por favor ingrese un tipo valido de opcion Europea: Call o Put."

        expected_payoff_at_maturity = np.mean(payoff)
        V = np.exp(-self.stochastic_process.rfrate * self.TTM) * expected_payoff_at_maturity
        print(f"El {self.type} con strike $ {round(self.K, 2)} y madurez a {round(self.TTM*252,4)} dias vale: \t $ {round(V, 2)}")
        return round(V,4)



class AnalisisDeSensibilidad_Simple():

    def __init__(self,x_inf,x_sup,dx,S_0,r,volatility,Lambda,beta,D,strike,TTM,op_type,jdNsim=2000,jdNsteps=252):
        '''
        Colocar 'x' en la variable a sensibilizar.
        Solo se puede sensibilizar una unica variable
        '''
        self.x_inf = x_inf
        self.x_sup = x_sup
        self.dx = dx

        self.S0 = S_0
        self.r = r
        self.volatility = volatility

        self.jump_rate = Lambda
        self.jump_intensity_mean = beta
        self.jump_intensity_sd = D

        self.strike = strike
        self.TTM = TTM
        self.op_type = op_type

        self.jdNsim = jdNsim    # Cant. de simulaciones del proceso de salto-difusion
        self.jdNsteps =jdNsteps # Cant. de pasos de 1 relizacion del proceso de salto-difusion


        self.eje_x = np.linspace(self.x_inf, self.x_sup, self.dx)  # Construccion del eje x, variable a sensiblizar
        self.sens_jd = self.sensibilizar_jd()
        self.sens_bs = self.sensibilizar_bs()

    def sensibilizar_jd(self):
        'Devuelve los valores de la opcion bajo salto-difusion a lo largo del eje x'
        # x: Eje de variable a sensibilizar, z_'[modelo]': Eje de precio de la opcion segun el modelo de valuacion
        # Calculo de precios mediante Salto-Difusion
        x = self.eje_x
        z_jd = np.array(
            [
                Option(
                    JumpDiffusionProcess(Nsim=self.jdNsim, Nsteps=self.jdNsteps,
                                         process_params={"S0": x if self.S0 == 'x' else self.S0,
                                                         "r": x if self.r == 'x' else self.r,
                                                         "volatility": x if self.volatility == 'x' else self.volatility,
                                                         "lamb": x if self.jump_rate == 'x' else self.jump_rate,
                                                         "beta": x if self.jump_intensity_mean == 'x' else self.jump_intensity_mean,
                                                         "D": x if self.jump_intensity_sd == 'x' else self.jump_intensity_sd,
                                                         "TTM": x if self.TTM == 'x' else self.TTM
                                                         }, seed=1),
                    option_params={"type": self.op_type,
                                   "strike": x if self.strike == 'x' else self.strike,
                                   "TTM": x if self.TTM == 'x' else self.TTM
                                   }).option_value()

                for x in np.ravel(x)])
        return z_jd

    def sensibilizar_bs(self):
        from BSModel import BS
        'Devuelve los valores de la opcion bajo Black-Scholes a lo largo del eje x'
        # x: Eje de variable a sensibilizar, z_'[modelo]': Eje de precio de la opcion segun el modelo de valuacion
        # Calculo de precios mediante Black-Scholes
        x = self.eje_x

        z_bs = np.array(
            [
                BS(self.op_type,                # type, stock, strike, maturity, interest, volatility, dividend)
                   stock = x if self.S0 == 'x' else self.S0,
                   strike = x if self.strike == 'x' else self.strike,
                   maturity = x if self.TTM == 'x' else self.TTM,
                   interest = x if self.r == 'x' else self.r,
                   volatility = x if self.volatility == 'x' else self.volatility,
                   dividend = 0
                   ).option_value() # Dividend = 0
                # type, stock, strike, maturity, interest, volatility, dividend)

                for x in np.ravel(x)])

        return z_bs

    def cambiar_resolucion_eje_x(self,multip_resolucion):
        'Aumenta la definicion para el suavizado de las curvas'
        x_new = np.linspace(self.x_inf, self.x_sup, multip_resolucion * self.dx)
        return x_new

    def suavizar_curva(self, multip_resolucion, z_values):
        x = self.eje_x
        x_new = self.cambiar_resolucion_eje_x(multip_resolucion)
        f = interpolate.interp1d(x, z_values, kind='cubic')
        z_smooth = f(x_new)
        return z_smooth

    def grafico_sensibilidad_simple(self,x_new,z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title):
        '''
        x_new: Eje x con la definicion aumentada
        z_jd_smooth: Valores de la opcion suavizados por interpolacion. Es la imagen de x_new
        x: Eje x original (lo usa la valuacion de BS ya que al ser suave no necesita ser suavizada)
        z_bs: Valores de la opcion calculados con el modelo de Black-Scholes
        '''

        plt.plot(x_new, z_jd_smooth, label='Salto-Difusión', color='blue')
        plt.plot(x_new, z_bs_smooth, label='Black-Scholes', color='red')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(x_axis_name, fontsize=20)
        plt.ylabel(y_axis_name, fontsize=20)
        plt.title(title, fontsize=20, fontweight="bold")
        plt.legend(fontsize=20)
        plt.show()






    ### GRAFICO DE SUPERFICIE DE SENSIBILIDAD ### - Funcion independiente
def superficie_sensibilidad(xxx,yyy,zzz,titulo,x_axis_name,y_axis_name,z_axis_name,contorno=False):
    '''Grafica la superficie  de sensibilidad del precio de la opcion
    Sus inputs son:
    - la terna (xxx,yyy,zzz)
    - los nombres de los ejes
    - el titulo del grafico
    - contorno (True o False) : Si es True, agrega las curvas de nivel en cada proyeccion bidimensional de la superficie
    donde:
    xxx, yyy es una grilla de tipo np.meshgrid, y zzz es una superficie interpolada de los valores de la opcion calculados (zz)
    '''
    # Plot surface
    print("Plotting surface ...")
    fig = plt.figure()
    fig.suptitle(titulo, fontsize=20)  # , fontsize=20
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xxx, yyy, zzz, rstride=1, cstride=1, alpha=0.75, cmap=cm.RdYlBu)
    ax.set_xlabel(x_axis_name,fontsize = 15)
    ax.set_ylabel(y_axis_name,fontsize = 15)
    ax.set_zlabel(z_axis_name,fontsize = 15)

    # Plot 3D contour

    if contorno == True:

        zzlevels = np.linspace(zz.min(), zz.max(), num=8, endpoint=True)
        xxlevels = np.linspace(xx.min(), xx.max(), num=8, endpoint=True)
        yylevels = np.linspace(yy.min(), yy.max(), num=8, endpoint=True)
        cset = ax.contourf(xx, yy, zz, zzlevels, zdir='z', offset=zz.min(),
                           cmap=cm.RdYlBu, linestyles='dashed')
        cset = ax.contourf(xx, yy, zz, xxlevels, zdir='x', offset=xx.min(),
                           cmap=cm.RdYlBu, linestyles='dashed')
        cset = ax.contourf(xx, yy, zz, yylevels, zdir='y', offset=yy.max(),
                           cmap=cm.RdYlBu, linestyles='dashed')

        for c in cset.collections:
            c.set_dashes([(0, (2.0, 2.0))])  # Dash contours

        plt.clabel(cset, fontsize=10, inline=1)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_zlim(zz.min(), zz.max())

    # ax.relim()
    # ax.autoscale_view(True,True,True)

    # Barra lateral de colores
    colbar = plt.colorbar(surf, shrink=1.0, extend='both', aspect=10)
    l, b, w, h = plt.gca().get_position().bounds
    ll, bb, ww, hh = colbar.ax.get_position().bounds
    colbar.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])

    # Mostrar grafico
    plt.show()


if __name__=='__main__':
    # Muestra de ejemplo de simulaciones del proceso y valuacion de un call
    my_jd_process = JumpDiffusionProcess(Nsim=100,Nsteps=252,process_params={"S0":100,"r":0.05,"volatility":0.1,"lamb":0.1,"beta":0.1,"D":0.1,"TTM":1},seed=1)
    my_jd_process.show_process()
    my_jd_process.final_histogram()

    my_option = Option(my_jd_process,option_params={"type":"call","strike":90,"TTM":1}).option_value()


    ###### ANALISIS DE SENSIBILIDAD SIMPLE Y COMPARACION DE MODELOS DE VALUACION ######

    from BSModel import BS

    print("\n")
    print("Analisis de sensibilidad <rapido>")
    print("\n\n")



    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S)
    # Parametros Constantes:
    S_0 = 100  # A sensibilizar
    r = 0.05
    volatility = 0.1
    Lambda = 0.25
    beta = 0.15
    D = 0.1
    TTM = 1
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    epsilon = 0.1  # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike * (1 - epsilon)
    S_sup = strike * (1 + epsilon)
    dx = 5  # Cant. de pasos en el eje x

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        S_inf, S_sup, dx, 'x', r, volatility, Lambda,
        beta, D, strike, TTM, op_type, jdNsim=50000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 5000
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = 'Precio del Acivo S'
    y_axis_name = 'Precio de la Opción'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type}, K = {strike}, TTM = {TTM}, r = {r}, " \
        f"$\sigma$ = {volatility}, $\lambda$ = {Lambda}, " \
        + rf"$\beta$ = {beta}, D = {D}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)



    ### PRECIO DE LA OPCION vs. TASA PROMEDIO DE OCURRENCIA DE SALTOS (Lambda)
    # Parametros Constantes:
    S_0 = 100
    r = 0.05
    volatility = 0.1
    Lambda = 0.25  # A sensibilizar
    beta = 0.15
    D = 0.1
    TTM = 1
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    Lambda_inf = 0.05
    Lambda_sup = 0.5
    dx = 5

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        Lambda_inf, Lambda_sup, dx, S_0, r, volatility, 'x',
        beta, D, strike, TTM, op_type, jdNsim=80000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 5000
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = 'Tasa de Ocurrencia Promedio de Saltos $\lambda$'
    y_axis_name = 'Precio de la Opción'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type},  $S_0$ = {S_0},  K = {strike},  TTM = {TTM}, " \
        f"r = {r},  $\sigma$ = {volatility}, " + rf" $\beta$ = {beta},  D = {D}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)



    ### PRECIO DE LA OPCION vs. INTENSIDAD MEDIA DE SALTOS (beta)
    # Parametros Constantes:
    S_0 = 100
    r = 0.05
    volatility = 0.1
    Lambda = 0.25
    beta = 0.1  # A sensibilizar
    D = 0.1
    TTM = 1
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    beta_inf = 0.01
    beta_sup = 0.7
    dx = 5

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        beta_inf, beta_sup, dx, S_0, r, volatility, Lambda,
        'x', D, strike, TTM, op_type, jdNsim=20000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 100
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = "Intensidad Media de Saltos " + rf"$\beta$"
    y_axis_name = 'Precio de la Opción'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type},  $S_0$ = {S_0},  K = {strike},  TTM = {TTM}, " \
        f"r = {r},  $\sigma$ = {volatility}, " + rf" $\lambda$ = {Lambda},  D = {D}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)



    ### PRECIO DE LA OPCION vs. DESVIO DE LA INTENSIDAD DE SALTOS (D)
    # Parametros Constantes:
    S_0 = 100
    r = 0.05
    volatility = 0.1
    Lambda = 0.2
    beta = 0.1
    D = 0.1  # A sensibilizar
    TTM = 1
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    D_inf = 0.01
    D_sup = 0.7
    dx = 5

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        D_inf, D_sup, dx, S_0, r, volatility, Lambda,
        beta, 'x', strike, TTM, op_type, jdNsim=15000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 5000
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = "Desvío de Intensidad de Saltos D"
    y_axis_name = 'Precio de la Opcion'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type},  $S_0$ = {S_0},  K = {strike},  TTM = {TTM}, " \
        f"r = {r},  $\sigma$ = {volatility}, " + rf" $\lambda$ = {Lambda},  $\beta$ = {beta}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)



    ### PRECIO DE LA OPCION vs. TIEMPO A EXPIRACION (TTM)
    # Parametros Constantes:
    S_0 = 100
    r = 0.05
    volatility = 0.1
    Lambda = 0.3
    beta = 0.2
    D = 0.1
    TTM = 1  # A sensibilizar
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    TTM_inf = 0.01
    TTM_sup = 2
    dx = 5

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        TTM_inf, TTM_sup, dx, S_0, r, volatility, Lambda,
        beta, D, strike, 'x', op_type, jdNsim=7000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 100
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = "Tiempo a Expiración TTM [años]"
    y_axis_name = 'Precio de la Opción'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type},  $S_0$ = {S_0},  K = {strike}, " \
        f"r = {r}, $\sigma$ = {volatility}" + rf" $\lambda$ = {Lambda}, $\beta$ = {beta}, D = {D}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)



    ### PRECIO DE LA OPCION vs. VOLATILIDAD (sigma)
    # Parametros Constantes:
    S_0 = 100
    r = 0.05
    volatility = 0.1  # A sensibilizar
    Lambda = 0.3
    beta = 0.2
    D = 0.1
    TTM = 1
    op_type = "call"
    strike = 90

    # Parametro a sensibilizar - Rango de perturbacion
    vol_inf = 0.01
    vol_sup = 0.8
    dx = 5

    # Calculo de curvas de sensibilidad
    sensibilidad_simple = AnalisisDeSensibilidad_Simple(
        vol_inf, vol_sup, dx, S_0, r, 'x', Lambda,
        beta, D, strike, TTM, op_type, jdNsim=25000, jdNsteps=252)

    z_jd = sensibilidad_simple.sensibilizar_jd()
    z_bs = sensibilidad_simple.sensibilizar_bs()

    # Suavizo las curvas de sensibilidad (interpolacion cubica + aumento definicion)
    multip_resolucion = 100
    x_new = sensibilidad_simple.cambiar_resolucion_eje_x(multip_resolucion)
    z_jd_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_jd)
    z_bs_smooth = sensibilidad_simple.suavizar_curva(multip_resolucion, z_bs)

    # Nombres de ejes y grafico
    x_axis_name = "Volatilidad $\sigma$"
    y_axis_name = 'Precio de la Opción'
    title = \
        f"Comparación de Valuaciones: Black-Scholes vs. Salto-Difusión \n " \
        f"Type = {op_type},  $S_0$ = {S_0},  K = {strike}, TTM = {TTM}, " \
        f"r = {r}, " + rf" $\lambda$ = {Lambda}, $\beta$ = {beta}, D = {D}"

    # Mostrar comparacion de modelos
    sensibilidad_simple.grafico_sensibilidad_simple(
        x_new, z_jd_smooth, z_bs_smooth, x_axis_name, y_axis_name, title)





    ######### SUPERFICIES DE SENSIBILIDAD #########

    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S) vs. INTENSIDAD MEDIA DE SALTOS (beta) ###
    # Parametros Constantes:
    r = 0.05
    volatility = 0.1
    Lambda = 0.5
    D = 0.1
    TTM = 1
    op_type = "call" #"put" para Put

    # Parametros a sensibilizar:
    # Definicion de la grilla
    dx, dy = 5, 5  # Cant. de pasos en los ejes de: S y beta

    # x: Eje de precios S, y: Eje de beta, zz: Eje de precio de la opcion
    strike = 90     # Precio de ejercicio de la opcion
    epsilon = 0.5   # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike*(1-epsilon)
    S_sup = strike*(1+epsilon)
    x = np.linspace(S_inf, S_sup, dx)       # Eje de precios S

    beta_inf = 0.01
    beta_sup = 0.7
    y = np.linspace(beta_inf, beta_sup, dy) # Eje de intensidad media de saltos (beta)

    xx, yy = np.meshgrid(x,y)   #Armo la grilla (dominio de R2 - soporte de la superficie)
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=3000, Nsteps=252,
                process_params={"S0": x, "r": r, "volatility": volatility, "lamb": Lambda,"beta": y, "D": D, "TTM": TTM}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": TTM}).option_value()

         for x, y in zip(np.ravel(xx), np.ravel(yy))]
                  )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 10
    f = interpolate.interp2d(x,y,zz,kind='cubic')
    x2 = np.linspace(S_inf, S_sup, multip_resolucion * dx)
    y2 = np.linspace(beta_inf, beta_sup, multip_resolucion*dy)
    xxx, yyy = np.meshgrid(x2,y2)
    zzz = f(x2,y2)

    # Nombre de ejes y grafico
    x_axis_name = "Precio del Activo S"
    y_axis_name = "Intensidad Media de Saltos "+r"$\beta$"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = 100,  K = {strike},  TTM = {TTM},  r = {r}  $\sigma$ = {volatility},  " \
             f"$\lambda$ = {Lambda},  D = {D}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)



    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S) vs. DESVIO DE INTENSIDAD DE SALTOS (D) ###
    # Parametros Constantes
    r = 0.05
    volatility = 0.1
    Lambda = 0.5
    beta = 0.1
    TTM = 1
    op_type = "call"

    # Parametros a sensibilizar:
    # Definicion de la grilla
    # Grid definition
    dx, dy = 5, 5  # Cant. de pasos en los ejes: S y D

    # x: Eje de precios S, y: Eje de D, zz: Eje de precio de la opcion
    strike = 90     # Precio de ejercicio de la opcion
    epsilon = 0.5   # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike*(1-epsilon)
    S_sup = strike*(1+epsilon)
    x = np.linspace(S_inf, S_sup, dx)   # Eje de precios S

    D_inf = 0.01
    D_sup = 1
    y = np.linspace(D_inf, D_sup, dy)   # Eje de desvios de intensidad de saltos D

    xx, yy = np.meshgrid(x, y)  # Armo la grilla
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=3000, Nsteps=252,
                                     process_params={"S0": x, "r": r, "volatility": volatility, "lamb": Lambda,
                                                     "beta": beta, "D": y, "TTM": TTM}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": TTM}).option_value()

            for x, y in zip(np.ravel(xx), np.ravel(yy))]
    )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 10
    f = interpolate.interp2d(x, y, zz, kind='cubic')
    x2 = np.linspace(S_inf, S_sup, multip_resolucion * dx)
    y2 = np.linspace(D_inf, D_sup, multip_resolucion * dy)
    xxx, yyy = np.meshgrid(x2, y2)
    zzz = f(x2, y2)

    # Nombre de ejes y grafico
    x_axis_name = "Precio del Activo S"
    y_axis_name = "Desvío de la Intensidad de Saltos D"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = 100,  K = {strike},  TTM = {TTM},  r = {r}  " \
             f"$\sigma$ = {volatility}, $\lambda$ = {Lambda},  "+rf"$\beta$ = {beta}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)



    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S) vs. TASA PROMEDIO DE OCURRENCIA DE SALTOS (Lambda) ###
    # Parametros Constantes
    r = 0.05
    volatility = 0.1
    beta = 0.25
    D = 0.2
    TTM = 1
    op_type = "call"

    # Parametros a sensibilizar:
    # Definicion de la grilla
    dx, dy = 5, 5  # Cant. de pasos en los ejes de: S y Lambda

    # x: Eje de precios S, y: Eje de Lambda, zz: Eje de precio de la opcion
    strike = 90     # Precio de ejercicio de la opcion
    epsilon = 0.5   # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike*(1-epsilon)
    S_sup = strike*(1+epsilon)
    x = np.linspace(S_inf,S_sup,dx) # Eje de precios S

    Lambda_inf = 0.01
    Lambda_sup = 1
    y = np.linspace(Lambda_inf, Lambda_sup, dy) # Eje de tasa de arribos Poisson (Lambda)

    xx, yy = np.meshgrid(x, y)  # Armo la grilla
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=5000, Nsteps=252,
                                     process_params={"S0": x, "r": r, "volatility": volatility, "lamb": y,
                                                     "beta": beta, "D": D, "TTM": TTM}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": TTM}).option_value()

            for x, y in zip(np.ravel(xx), np.ravel(yy))]
    )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 10
    f = interpolate.interp2d(x, y, zz, kind='cubic')
    x2 = np.linspace(S_inf, S_sup, multip_resolucion * dx)
    y2 = np.linspace(Lambda_inf, Lambda_sup, multip_resolucion * dy)
    xxx, yyy = np.meshgrid(x2, y2)
    zzz = f(x2, y2)

    # Nombre de ejes y graficos
    x_axis_name = "Precio del Activo S"
    y_axis_name = "Tasa de Promedio de Ocurrencia de Saltos $\lambda$"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = 100,  K = {strike},  TTM = {TTM},  r = {r}  " \
             f"$\sigma$ = {volatility},  D = {D},  " + rf"$\beta$ = {beta}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)



    ### PRECIO DE LA OPCION vs. TASA PROMEDIO DE OCURRENCIA DE SALTOS (Lambda) vs. INTENSIDAD MEDIA DE SALTOS (beta) ###
    # Parametros Constantes:
    S_0 = 100 # Precio del activo observado HOY (t=0)
    r = 0.05
    volatility = 0.1
    D = 0.1
    strike = 90
    TTM = 1
    op_type = "call"

    # Parametros a sensibilizar:
    # Definicion de la grilla
    dx, dy = 5, 5  # Cant. de pasos en los ejes de: beta y Lambda

    # x: Eje de beta, y: Eje de Lambda, zz: Eje de precio de la opcion
    beta_inf = 0.01
    beta_sup = 0.6
    x = np.linspace(beta_inf,beta_sup)  # Eje de beta

    Lambda_inf = 0.01
    Lambda_sup = 0.6
    y = np.linspace(Lambda_inf,Lambda_sup,dy)   # Eje de Lambda

    xx, yy = np.meshgrid(x, y) # Armo la grilla
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=15000, Nsteps=252,
                                     process_params={"S0": S_0, "r": r, "volatility": volatility, "lamb": y,
                                                     "beta": x, "D": D, "TTM": TTM}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": TTM}).option_value()

            for x, y in zip(np.ravel(xx), np.ravel(yy))]
    )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 20
    f = interpolate.interp2d(x, y, zz, kind='cubic')
    x2 = np.linspace(beta_inf, beta_sup, multip_resolucion * dx)
    y2 = np.linspace(Lambda_inf, Lambda_sup, multip_resolucion * dy)
    xxx, yyy = np.meshgrid(x2, y2)
    zzz = f(x2, y2)

    # Nombre de ejes y graficos
    x_axis_name = "Intensidad Media de Saltos "+r"$\beta$"
    y_axis_name = "Tasa de Promedio de Ocurrencia de Saltos $\lambda$"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = {S_0},  K = {strike},  TTM = {TTM},  " \
             f"r = {r}  $\sigma$ = {volatility},  D = {D}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)



    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S) vs. TIME TO MATURITY (TTM) ###
    # Parametros Constantes:
    r = 0.05
    volatility = 0.1
    Lambda = 0.5
    beta = 0.1
    D = 0.1
    op_type = "call"

    # Parametros a sensibilizar:
    # Definicion de la grilla
    dx, dy = 5, 5  # Cant. de pasos en los ejes de: S y TTM

    # x: Eje de precios S, y: Eje de tiempo a expiracion TTM, zz: Eje de precio de la opcion
    strike = 90     # Precio de ejercicio de la opcion
    epsilon = 0.5   # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike*(1-epsilon)
    S_sup = strike*(1+epsilon)
    x = np.linspace(S_inf,S_sup,dx) # Eje de precios S

    TTM_inf = 0.01
    TTM_sup = 2
    y = np.linspace(TTM_inf,TTM_sup,dx) # Eje de tiempo a expiracion TTM

    xx, yy = np.meshgrid(x, y)  # Armo la grilla
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=20000, Nsteps=252,
                                     process_params={"S0": x, "r": r, "volatility": volatility, "lamb": Lambda,
                                                     "beta": beta, "D": D, "TTM": y}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": y}).option_value()

            for x, y in zip(np.ravel(xx), np.ravel(yy))]
    )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 15
    f = interpolate.interp2d(x, y, zz, kind='cubic')
    x2 = np.linspace(S_inf, S_sup, multip_resolucion * dx)
    y2 = np.linspace(TTM_inf, TTM_sup, multip_resolucion * dy)
    xxx, yyy = np.meshgrid(x2, y2)
    zzz = f(x2, y2)

    # Nombres de ejes y graficos
    x_axis_name = "Precio del Activo S"
    y_axis_name = "Tiempo a Expiración TTM [años]"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = 100,  K = {strike},  r = {r}  " \
             f"$\sigma$ = {volatility},  $\lambda$ = {Lambda},  D = {D}, "+rf"$\beta$ = {beta}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)



    ### PRECIO DE LA OPCION vs. PRECIO DEL ACTIVO (S) vs. VOLATILIDAD (sigma) ###
    # Parametros Constantes:
    r = 0.05
    Lambda = 0.5
    beta = 0.1
    D = 0.1
    TTM = 1
    op_type = "call"

    # Parametros a sensibilizar:
    # Definicion de la grilla
    dx, dy = 5, 5  # Cant. de pases en los ejes de: S y sigma

    # x: Eje de precios S, y: Eje de volatilidades (sigma), zz: Eje de precio de la opcion
    strike = 90     # Precio de ejercicio
    epsilon = 0.5   # Rango % alrededor del precio de ejercicio que miro el precio del activo S
    S_inf = strike*(1-epsilon)
    S_sup = strike*(1+epsilon)
    x = np.linspace(S_inf,S_sup, dx)    # Eje de precios S

    vol_inf = 0.01
    vol_sup = 0.8
    y = np.linspace(vol_inf, vol_sup, dy)   # Eje de volatilidades (sigma)

    xx, yy = np.meshgrid(x, y)  # Armo la grilla
    print("Calculando superficie de precios ...")
    zz = np.array(
        [
            Option(
                JumpDiffusionProcess(Nsim=20000, Nsteps=252,
                                     process_params={"S0": x, "r": r, "volatility": y, "lamb": Lambda,
                                                     "beta": beta, "D": D, "TTM": TTM}, seed=1),
                option_params={"type": op_type, "strike": strike, "TTM": TTM}).option_value()

            for x, y in zip(np.ravel(xx), np.ravel(yy))]
    )

    zz = zz.reshape(xx.shape)

    # Suavizado de superficie de precios (interpolacion + aumento de definicion)
    multip_resolucion = 15
    f = interpolate.interp2d(x, y, zz, kind='cubic')
    x2 = np.linspace(S_inf, S_sup, multip_resolucion * dx)
    y2 = np.linspace(vol_inf, vol_sup, multip_resolucion * dy)
    xxx, yyy = np.meshgrid(x2, y2)
    zzz = f(x2, y2)

    # Nombre de ejes y grafico
    x_axis_name = "Precio del Activo S"
    y_axis_name = "Volatilidad $\sigma$"
    z_axis_name = "Precio de la Opción"
    titulo = f"Precio de la Opción bajo Salto-Difusión \n $S_0$ = 100,  K = {strike},  r = {r}  TTM = {TTM},  " \
             f"$\lambda$ = {Lambda},  D = {D}, " + rf"$\beta$ = {beta}"

    # Mostrar superficie de sensibilidad
    superficie_sensibilidad(xxx, yyy, zzz, titulo, x_axis_name, y_axis_name, z_axis_name)














