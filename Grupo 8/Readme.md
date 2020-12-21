# **OPTIMIZACIÓN DE PORTAFOLIOS - BACKTESTING**

###**"La diversificación es la protección contra la ignorancia"**

Que mejor manera de comenzar con este trabajo final citando al gran
Warren Buffet, con una frase más que polémica y contradiciendo a todos
aquellos grandes inversores (ej: Peter Lynch) que tienen a la
diversificación como un estilo de vida.

Lo cierto es, que lo que afirma Buffet, es correcto. Los inversores no
profesionales (e incluso varios profesionales) no son infalibles ni
tienen un track record perfecto a la hora de encontrar una sola empresa
en la cual poner todo su capital, y deben (debemos) diversificar
nuestras carteras para tratar de maximizar el rendimiento y minimizar el
riesgo.

Comenzando con el marco teórico, partimos de la publicación de la Teoría
de Selección de Carteras de Markowitz (1952). El autor publica un modelo
cuyo objetivo consiste en encontrar la cartera de inversión óptima para
cada inversor en términos de rentabilidad y riesgo.

Los principales supuestos de su modelo fueron tres:

• El rendimiento de cualquier título o cartera es descripto por una
variable aleatoria subjetiva, cuya distribución de probabilidad, para el
periodo de referencia, es conocida por el inversor. El rendimiento del
título o cartera será medido a través de su esperanza matemática.

• El riesgo de un título o cartera viene medido por la varianza (o
desviación típica) de la variable aleatoria representativa de su
rendimiento.

• El inversor preferirá aquellos activos financieros que tengan un mayor
rendimiento para un riesgo dado, o un menor riesgo para un rendimiento
conocido.

A partir de la Teoría de Selección de Carteras de Markowitz, surge el
modelo CAPM (Capital Asset Pricing Model) desarrollado por William
Sharpe, John Lintner y Jan Mossin.

El objetivo principal del CAPM es estimar la rentabilidad de activos
financieros o carteras con base en su riesgo y encontrar un indicador
que represente el riesgo de dicho activo o cartera con respecto del
mercado. Este indicador es llamado BETA (β).

Para la construcción del Modelo CAPM se asumieron los siguientes
supuestos:

• Los inversores son personas racionales, los cuales buscan maximizar la
rentabilidad y minimizar el riesgo. Esto quiere decir que se comportan
bajo el modelo de media-varianza desarrollado por Markowitz.

• Los inversores solo deben preocuparse por el riesgo sistemático, es
decir, el riesgo que depende del riesgo de mercado. Puesto que tienen la
posibilidad de diversificar sus activos al formar sus carteras.

• El mercado es eficiente, por tanto, no existe asimetría de
información.

• Ningún inversor tiene la capacidad de influir en el precio de los
activos, esto quiere decir que es un mercado de competencia perfecta.

• Existe una tasa libre de riesgo (Rf) a la cual los inversores pueden
endeudarse o prestar fondos.

• Los costes de transacción, así como los impuestos son iguales para
todos los inversores.

El modelo CAPM se formula de la siguiente manera:

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image1.png) 

Donde 𝐸(𝑅𝑖) es la rentabilidad esperada del activo estudiado, 𝑅𝑓 es la
rentabilidad del activo libre de riesgo para el mercado en el que se
encuentra el activo, 𝛽 es el coeficiente de la prima de riesgo de
mercado, el cual mide la sensibilidad de la rentabilidad de un activo
con respecto a cambios en la rentabilidad de mercado y (𝐸(𝑅𝑚) − 𝑅𝑓) es
la prima de riesgo de mercado, la cual viene dada por la diferencia
entre la rentabilidad esperada de mercado y la rentabilidad del activo
libre de riesgo.

Lo más importante de este modelo es el factor 𝛽, puesto que es el riesgo
sistemático y depende del riesgo del mercado donde se negocie el activo
objeto de estudio. Esto quiere decir que dicho riesgo no puede ser
mitigado mediante la diversificación en la cartera.

Así pues y según el modelo CAPM, un inversor nunca debe asumir un riesgo
diversificable, ya que solo obtendría mayores rentabilidades por asumir
el riesgo de mercado.

Este factor puede tomar diferentes valores como pueden ser:

0: se espera que la rentabilidad del activo sea similar a la tasa libre
de riesgo.

1: Indica que la rentabilidad del activo se mueve igual que la
rentabilidad del mercado.

\>1: Indica que la rentabilidad del activo es más volátil que la
rentabilidad del mercado.

El factor 𝛽 relaciona la covarianza entre la rentabilidad del activo
objeto de estudio y la rentabilidad del mercado con la varianza del
mercado, la cual representa la volatilidad de dicho mercado. A raíz de
esto, este factor puede estimarse mediante la siguiente expresión:

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image2.png)

Es menester destacar que, en dichos modelos, la contribución de cada
activo al riesgo (desvío estándar) del portafolio no depende únicamente
de la ponderación (que tiene en el portafolio) y de su riesgo individual
(volatilidad), sino también de cómo se correlaciona (como covaría) con
los demás activos financieros.

Para evaluar a distintos fondos, portafolios o vehículos de inversión,
muchos inversores a lo largo del mundo utilizan ratios estadísticos que
miden la rentabilidad obtenida en función del riesgo asumido.

El Sharpe Ratio fue desarrollado por el Premio Nóbel William Sharpe de
la Universidad de Stanford. Mide numéricamente la relación Rentabilidad
/ Volatilidad Histórica (desviación estándar) de un Fondo de Inversión.
Se calcula dividiendo la rentabilidad de un fondo menos la tasa de
interés sin riesgo entre la volatilidad o desviación estándar de esa
rentabilidad en el mismo periodo.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image3.png)

Otro ratio muy popular, y primo hermano del Sharpe, es el ratio de
Sortino. Como se observa, la fórmula es prácticamente igual, con la
salvedad de que utiliza como medida de ajuste del riesgo únicamente la
desviación estándar generada por los rendimientos negativos de la
cartera.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image4.png)

###**Metodología y datos utilizados para el trabajo**

El objetivo del trabajo realizado fue comparar una estrategia activa de
gestión de carteras basadas en el momentum de las acciones del SP500
contra un benchmark definido por una estrategia de buy & hold del ETF
SPY en el período 2000-2020.

Definimos momentum como el rendimiento obtenido en el último período de
tiempo considerado. Al trabajar con rebalanceos trimestrales, como es
costumbre en la industria financiera de portfolios managers, cada
portfolio estará aleatoriamente integrada con los papeles de mejores
rendimientos en el trimestre pasado.

La estrategia de gestión activa se basó en los siguientes lineamientos:

-   5 activos.

-   Cartera 100% invertida el total del tiempo. No tiene posición cash.

-   Rebalanceo trimestral en base a momentum.

-   No hay posibilidad de ir Short.

-   Peso máximo de cada activo = 50%.

-   Peso mínimo de cada activo = 5%.

Los datos fueron extraídos desde la librería Yahoo Finance.

La elaboración de los distintos portafolios al azar se realizó mediante
una simulación del método de Montecarlo, a través de la generación de
variables aleatorias.

Utilizamos un criterio de selección de portafolios de mejor a peor en
base a un ratio de sharpe simplificado, donde tomamos retornos sobre
volatilidad.

Al finalizar el Script, comparamos ambas inversiones con un reporte
estadístico completo elaborado por la librería quantstats de Python,
analizando ratios de sharpe, sortino, cagr, volatilidad, drawdowns,
entre otros aspectos que consideramos críticos.

Adicionalmente, al observar ciertos parámetros que un inversor
tradicional no estaría dispuesto a tolerar, se propone una cartera
alternativa, pensando en un perfil conservador aplicando una simil ley
Pareto, donde colocamos el 80% de nuestro portafolio en tasa libre de
riesgo (2.5% anual) y el 20% en la estrategia de gestión activa.

###**Trabajo realizado**

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image5.png)

Como primeros pasos, importamos las
librerías de Python que utilizaremos y traemos los 500 tickers del
S&P500, excepto Berkshire Hataway y Brown-Forman Corporation debido a
cuestiones de sintaxis en sus tickers.

Traemos los datos desde Yahoo Finance desde el 1 de Enero de 2000 hasta
la actualidad. Elegimos comenzar desde principios de décadas ya que
tenemos dos fuertes crisis conocidas a nivel mundial (Burbuja .COM y
Sub-Prime).

Para evitar trabajar con tickers con poca presencia en el índice,
solamente utilizamos aquellos que tengan mas de 1250 datos (250 ruedas
búrsatiles por año, trabajamos con papeles que hayan estado presentes en
5 de los 20 años del período de análisis).

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image6.png)

Tal cual mencionamos anteriormente, las carteras estarán compuestas por
5 activos con ponderaciones mínimas individuales de 5% y máximas del
50%. Utilizamos la distribución de Dirichlet para calcular
aleatoriamente estas ponderaciones

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image7.png)

Ahora bien, definimos la función de optimización y ordenamos los
portafolios obtenidos en base a un ratio de sharpe modificado (Simple,
solo retorno/volatilidad)

De los tickers del SP500, toma de a 5 al azar (muestra).

Tiro 100 valores posibles de ponderaciones al azar, y tomo la primera.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image8.png)

Definimos los trimestres en el período de tiempo a analizar

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image9.png)

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image10.png)

Realizamos un DataFrame para obtener los
retornos logarítmicos de cada activo del sp500

Utilizamos Montecarlo para la generación de variables aleatorias, iterar
y armar 25000 combinaciones al azar, las cuáles ordenamos de acuerdo a
nuestro Ratio de Sharpe Simplificado.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image11.png)

Observamos cuales fueron los tickers que más se repitieron, para armar
portafolios con los papeles de mejores retornos.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image12.png)

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image13.png)

Ahora bien, a continuación, nuevamente con
Montecarlo, generamos variables aleatorias a partir de una cantidad
inicial de 1000 combinaciones, dónde luego el algoritmo tomará el mejor
tercio para iterar otras 500 combinaciones, y así sucesivamente en 8
sesiones de entrenamiento.

A priori, se observa que los ratios de Sharpe de la primera iteración de
25.000 combinaciones aleatorias, aumentaron considerablemente.

Ahora bien, debemos darle el rebalanceo trimestral a nuestra estrategia,
entonces definimos una función de los mejores 10 portafolios para
insertarlo en la función de optimización para los 83 trimestres. Corro
83 veces lo mismo que hicimos antes, pero con una cantidad inicial de
500 portafolios y 5 sesiones de entrenamiento.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image14.png)

Trimestre a trimestre va a ir generando
aleatoriamente los 10 portafolios ideales.

Continuamos definiendo dos listas, una con los 83 portafolios top ten, y
otra con las 83 ponderaciones. Luego, observamos en un dataframe como
rindieron cada uno de esos 10 mejores portafolios.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image15.png)


Comparamos con el Benchmark definido (SPY):

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image16.png)

Definimos una estrategia con el promedio de los TOP10, para su
comparación contra el benchmark definido, utilizando el reporte
estadístico de quantstats

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image17.png)


![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image18.png)


![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image19.png)


A priori, observamos que, si bien el retorno acumulado y el CAGR son
extraordinarios, el ratio de Sharpe y Sortino son mejores que el SPY,
hay una alarma en dos ítems:

### **El máximo drawdown y días en drawdown** hace difícil de ver a esta
alternativa como atractiva para un inversor promedio.

Como una potencial solución, decidimos aplicar una símil ley Pareto a
nuestra cartera. Invertiremos el 20% en riesgo (gestión activa de
cartera en base a momentum) y el 80% a una supuesta tasa libre de riesgo
del 2.5% anual.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image20.png)

Corremos el reporte estadístico completo, y observamos que mejoran considerablemente los problemas que teníamos anteriormente (Drawdown).

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image21.png)

Además, tanto el Sharpe como el Sortino, aumentan.

![Alt text](https://github.com/guillermobr/ucema_tp_opt_portafolio/blob/main/media/image22.png)


### **CONCLUSIONES**

La estrategia de momentum utilizada es una apuesta sobre los
rendimientos pasados, aplicada posicionando la cartera LONG en los
grandes ganadores y sin ir short en los perdedores(otra alternativa). Es
notable el rendimiento acumulado superando ampliamente al benchmark,
principalmente explicado por la tasa de crecimiento anual compuesta
(CAGR).

El ratio de Sharpe y el de Sortino son superiores, es decir que
obtenemos más rentabilidad por unidad de riesgo asumida.

Ahora bien, no es todo color de rosas ni hemos descubierto la panacea.
Hay varios puntos que hemos discutido, pensando como un inversor
promedio:

**MAX Drawdown:** Atraeríamos algún inversor sabiendo que el portafolio
estuvo 68% abajo en algún momento? Probablemente no

**Longest Drawdown Days:** En línea con lo anterior, 1723 días para
recuperar el drawdown parece un tanto elevado.

Es por ello, que decidimos proponer como solución potencial una cartera
definida como EstrategiaPareto, en la cuál la cartera estará invertida
en un 80% a una tasa libre de riesgo al 2.5% anual y el restante 20% en
la estrategia de gestión activa en base al momentum.

Aplicando EstrategiaPareto, obtenemos resultados satisfactorios, que
denotan un rendimiento acumulado muy similar al benchmark, pero logramos
reducir sustancialmente la volatilidad y el drawdown del mismo.
