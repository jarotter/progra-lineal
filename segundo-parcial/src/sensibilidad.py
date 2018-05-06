# -*- coding: utf-8 -*-
from src import respuestas as rp
import numpy as np
import scipy.linalg as linalg

def analisis_de_sensibilidad_min(A, b, c, resp, min=True):
    """Realiza análisis de sensibilidad sobre la función y las restricciones.

    A problemas de minimización, si min=True, o de maximización si min=False.

    Parameters
    ----------
    A: numpy array de (m x n)
        La matriz con las restricciones sobre las soluciones factibles.
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles.
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector.
    resp: Respuesta

    Returns
    -------
    s: Sensibilidad
        Ver documentación de clase.
    """

    n = len(c)
    m = len(b)
    B = resp.basicas
    N = resp.no_basicas
    x = resp.optimal_point
    x_B = x[B]
    #print(x_B)
    A_B = np.column_stack([A[:,i] for i in B])
    A_N = np.column_stack([A[:,i] for i in N])
    A_B_inv = linalg.inv(A_B)

    gammas = np.zeros((n,2))
    lambdas = -np.dot(c[B].T, A_B_inv)
    betas = np.zeros((m,2))

    #Cotas para b
    for j in range(m):
        try:
            betas[j,0] = np.max([-x_B[i]/A_B_inv[i,j] for i in range(m) if A_B_inv[i,j]>0]) + b[j]
        except ValueError:
            betas[j,0] = -np.inf

        try:
            betas[j,1] = np.min([-x_B[i]/A_B_inv[i,j] for i in range(m) if A_B_inv[i,j]<0]) + b[j]
        except ValueError:
            betas[j,1] = np.inf

    #Cotas para c en variables no-básicas
    H = np.dot(A_B_inv, A_N)
    r_N = np.dot(c[B].T, H) - c[N]
    gammas[N,0] = c[N] + r_N
    gammas[N,1] = np.inf

    #Cotas para c en variables básicas
    for jj in B:
        j = np.where(B==jj)
        try:
            gammas[jj,0] = np.max([-r_N[i]/H[j,i] for i in range(n-m) if H[j,i] < 0]) + c[jj]
        except ValueError:
            gammas[jj,0] = -np.Inf
        try:
            gammas[jj,1] = np.min([-r_N[i]/H[j,i] for i in range(n-m) if H[j,i] > 0]) + c[jj]
        except ValueError:
            gammas[jj,1] = np.inf


    return(rp.Sensibilidad(lambdas, gammas, betas, resp.n_orig, m))

def analisis_de_sensibilidad_max(A,b,c,resp):
    """Realiza análisis de sensibilidad sobre la función y las restricciones.

    A problemas de la forma

        máx. c^Tx
            s.a Ax <= b
                 x >= 0
                 b >= 0

    Parameters
    ----------
    A: numpy array de (m x n)
        La matriz con las restricciones sobre las soluciones factibles.
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles.
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector.
    resp: Respuesta
        La respuesta al problema.

    Returns
    -------
    s: Sensibilidad
        Ver documentación de clase.
    """

    s = analisis_de_sensibilidad_min(A, b, c, resp)
    new_gammas = np.zeros_like(s.gammas)
    new_gammas[:,0] = -s.gammas[:,1]
    new_gammas[:,1] = -s.gammas[:,0]
    s.gammas = new_gammas

    return(s)
