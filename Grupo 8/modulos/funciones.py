import random
import sys

import numpy as np
import pandas as pd
import tqdm


def simular(data, tickers, q, n_stocks=5, w_min=0.05, w_max=0.50):
    datos = []
    with tqdm.tqdm(total=q, file=sys.stdout) as pbar:
        for i in range(q):
            pbar.update()

            muestra = data[random.sample(tickers, n_stocks)]

            ponds = np.random.dirichlet(np.ones(n_stocks), 100)
            pond = np.array([w for w in ponds if (w.min() > w_min) & (w.max() < w_max)][0])

            if not muestra.empty:
                r = dict()
                r['activos'] = list(muestra.columns)
                r['weights'] = pond.round(5)
                r['retorno'] = np.sum((muestra.mean() * pond * 252))
                r['volatilidad'] = np.sqrt(np.dot(pond, np.dot(muestra.cov() * 252, pond)))
                r['Sharpe Simple'] = round(r['retorno'] / r['volatilidad'], 5)
                datos.append(r)

    return pd.DataFrame(datos).sort_values('Sharpe Simple', ascending=False)


def top10(ret_log, lista_tickers, q_inicial=1000, rondas=10, n_inicial=5):
    portfolios = simular(ret_log, lista_tickers, q_inicial, n_stocks=n_inicial, w_min=0.05, w_max=0.50)
    best = pd.DataFrame()
    for i in range(rondas):
        qsim = int(q_inicial / (i + 2))
        qtop = qsim // 3

        top = portfolios.iloc[: qtop]
        lista_tickers = list(np.array(top.activos.apply(pd.Series).stack()))
        portfolios = simular(ret_log, lista_tickers, qsim)
        best = pd.concat([best, portfolios.iloc[:10]])

    top10 = best.sort_values('Sharpe Simple', ascending=False).head(10)
    return top10
