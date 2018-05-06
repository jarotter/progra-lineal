from math import log, exp
import numpy as np
from numpy.random import rand, normal
from numpy import round, int, abs, array
import simplex as simplex
import matplotlib.pyplot as plt

def test_simplex(num_tests = 50):
    """
    Prueba el método simplex que implementamos con una simulación que
    elige arbitratriamente m, n, A, b y c siguiendo una normal.

    Parámetros
    ----------
    num_tests : entero
        Número de simulaciones a realizar

    Regresa
    -------
    datos: vector de vectores
        Vector con elementos de la forma [n, m, iteraciones, estatus], donde
            n: entero
                Número de variables
            m: entero
                Número de restricciones
            iteraciones: entero
                Número de iteraciones
            estatus: bool
                False si el problema era acotado y hubo solución
                True si el problema fue no acotado


    """
    datos = []

    for i in range(num_tests):

        m = int(round(10*exp(log(20)*rand())))
        n = int(round(10*exp(log(20)*rand())))
        sigma = 100

        A = round(sigma*np.random.normal(0,1,(m,n)))

        b = round(sigma*abs(normal(0,1,(m,1))))
        b = b[:,0]

        c = round(sigma*normal(0,1,(n,1)))
        c = c[:,0]

        res = simplex.fase_ii(A,b,c, A.shape[1])

        iteraciones = res['Número de iteraciones']
        estatus = bool(res['Estatus'])

        datos.append([n,m,iteraciones, estatus])

    return datos

def make_plot(num_tests):
    """
    Hace la gráfica del número de iteraciones que simplex tomó
    en resolver una simulación contra el mínimo entre sus
    ecuaciones y sus restricciones.

    Guarda la gráfica como logplot.pdf en el directorio activo.

    Parámetros
    ----------
    num_tests: entero
        Número de simulaciones a realizar

    """

    datos = test_simplex(num_tests)

    all_n = [a[0] for a in datos]
    all_m = [a[1] for a in datos]
    min_m_n = np.log10(np.minimum(all_n, all_m))
    num_it = np.log10([a[2] for a in datos])
    is_bounded = [a[3] for a in datos]

    fit = np.polyfit(min_m_n, num_it, 1)
    m = fit[0]
    b = fit[1]
    print(m,b)

    line = m*min_m_n + b

    ax = plt.gca()
    ax.scatter(min_m_n[is_bounded], num_it[is_bounded], alpha=0.5,
                color='r', label="unbounded")
    ax.scatter(min_m_n[np.logical_not(is_bounded)],
                num_it[np.logical_not(is_bounded)], alpha=0.5, color='b', label = 'bounded')
    ax.plot(min_m_n,line)
    ax.legend()
    plt.xlabel('mín{n,m}')
    plt.ylabel('número iteraciones')
    plt.title('Iteraciones contra mín{m,n}')
    plt.savefig('logplot.pdf', format = 'pdf')

    return

make_plot(3)
