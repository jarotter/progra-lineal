# -*- coding: utf-8 -*-
import numpy as np

def estandarizar(A,b,c, leq=True):
    """Pasa el problema a forma estándar añadiendo variables de holgura.

    Parameters
    ----------
    A: numpy array (m x n)
        La matriz con las restricciones sobre las soluciones factibles
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector
    leq: boolean
        Si es True, el problema tiene todas sus restricciones <=; si
        es False, las tiene de la forma >=.

    Returns
    -------
    A: numpy array (m x (n+m))
        La matriz aumentada con una identidad de variables de holgura. Es
        decir, queda de la forma [A|1]
    c: numpy array ((n+m) x 1)
        La función objetivo, aumentada con ceros en las variables de holgura.
    """
    c = np.hstack((c,np.zeros(len(b))))
    A = np.hstack((A,np.eye(len(b)))) if leq else np.hstack((A,-np.eye(len(b))))
    return(A, c)

def blands_rule(r_N, N):
    """Elige la variable de entrada según la regla de Bland.

    Parameters
    ----------
    r_N: numpy array ((n-m) x 1)
        Los costos reducidos (o el lado derecho en Simplex Dual)
    N: numpy array ((n-m) x 1)
        Las variables no básicas

    Returns
    -------
    entrada: int
        El índice de la primera variable no básica con r>0
    """
    entrada = -1
    for i in range(len(r_N)):
        if r_N[i] > 0:
            entrada = N[i]
            break
    return(entrada)

def lexicographic(h, H_e, B):
    """Usa comparación lexicográfica para elegir una variable de salida.

    Parameters
    ----------
    h: numpy array (b x 1)
        El valor actual del lado derecho
    H_e: numpy array (m x 1)
        La columna de A_B^{-1}A_N que corresponde a la variable de entrada
    B: numpy array (m x 1)
        La base actual.

    Returns
    -------
    salida: int
        La variable de salida
    """
    #Para que las comparaciones funcionen sin importar los valores iniciales
    cociente = np.Inf
    indice_actual = -1

    for i in range(len(h)):
        if H_e[i]>0 and h[i]/H_e[i] < cociente and B[i] > indice_actual:
            cociente = h[i]/H_e[i]
            salida = B[i]
    #Usando la tercera condición, si hay empate nos quedamos con el de
    #menor índice
    return(salida)

def update(B,N, entrada, salida):
    """Regresa las nuevas variables básicas y no básicas.

    Parameters
    ----------
    B: numpy array (m x 1)
        Las variables básicas actuales.
    N: numpy array ((n-m) x 1)
        Las variables no-básicas actuales.
    entrada: int
        La variable que entrará a B.
    salida: int
        La variable que saldrá de B.

    Returns
    -------
    B, N
    """
    for i in range(len(B)):
        if B[i] == salida:
            B[i] = entrada

    for i in range(len(N)):
        if N[i] == entrada:
            N[i] = salida

    return(B,N)
