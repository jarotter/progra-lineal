# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as linalg
from src import simplex_utilities as utils
from src import respuestas as rp
from src import sensibilidad as ss

def _simplex_one_step(A,b,c,B,N, n_orig):
    """ Actualiza B y N.

    Actualiza las variables básicas (B) y no-básicas (N) de la solución
    actual al problema.

    Parameters
    ----------
    A: matriz de mxn
        La matriz con las restricciones sobre las soluciones factibles
    b: vector de m
        El vector de restricciones sobre las soluciones factibles
    c : vector de n
        El funcional lineal a optimizar, en forma de vector
    B: vector
        Los índices de las variables básicas
    N: vector
        Los índices de las variables no básicas

    Returns
    -------
    r: Respuesta
        Ver documentación de clase.
    """

    c_B = np.array([c[i] for i in B])
    c_N = np.array([c[i] for i in N])

    #Usamos column_stack porque numpy lo hace transpuesto
    A_N = np.column_stack([A[:,i] for i in N])
    A_B = np.column_stack([A[:,i] for i in B])

    lambda_ = linalg.solve(np.transpose(A_B), c_B)
    r_N = np.dot(lambda_, A_N) - c_N
    h = linalg.solve(A_B,b)

    if all(r_N <= 0):
        #Encontramos la solución óptima
        return(rp.Respuesta(flag=0, h=h, z0=np.dot(lambda_,b), B=B, N=N, n_orig=n_orig))

    #Usamos la regla de Bland para elegir variables
    entrada = utils.blands_rule(r_N, N)

    H_e = linalg.solve(A_B, A[:,entrada])

    if(max(H_e)<=0):
        #El problema no es acotado
        return(rp.Respuesta(flag=1, h=h, var_desc=entrada, d=H_e, z0=-np.Inf, n_orig=n_orig))

    #Salida
    salida = utils.lexicographic(h, H_e, B)

    B,N = utils.update(B,N, entrada, salida)

    return(rp.Respuesta(flag=2, B=B, N=N))

def _dual_simplex_one_step(A,b,c,B,N, n_orig):
    """ Actualiza B y N.

    Actualiza las variables básicas (B) y no-básicas (N) de la solución
    actual al problema primal.

    Parameters
    ----------
    A: matriz de mxn
        La matriz con las restricciones sobre las soluciones factibles
    b: vector de m
        El vector de restricciones sobre las soluciones factibles
    c : vector de n
        El funcional lineal a optimizar, en forma de vector
    B: vector
        Los índices de las variables básicas
    N: vector
        Los índices de las variables no básicas

    Returns
    -------
    r: Respuesta
        Ver documentación de clase.
    """

    A_B = np.column_stack([A[:,i] for i in B])
    A_N = np.column_stack([A[:,i] for i in N])
    x_B = linalg.solve(A_B,b)
    m = len(B)

    if all(x_B >=0):
        #Encontramos una solución primal-factible
        return(rp.Respuesta(flag=0, h=x_B, B=B, N=N, z0=np.dot(c[B], x_B), n_orig=n_orig))

    salida = utils.blands_rule(-x_B, B)
    s = np.where(B == salida)

    e_s = np.eye(m)[s].flatten().T
    delta_l = linalg.solve(A_B, e_s)
    H_s = np.dot((delta_l).T, A_N)

    if all(H_s>=0):
        #El problema no tiene solución factible
        return(rp.Respuesta(flag=-1))

    y = linalg.solve(A_B, c[B].T)
    r_N = np.dot(y,A_N) - c[N]
    entrada = utils.lexicographic(-r_N, -H_s, N)

    B,N = utils.update(B, N, entrada, salida)

    return(rp.Respuesta(flag=2, B=B, N=N))


def fase_ii_min(A,b,c, n_orig, B=None, N=None, dual=False):
    """Segunda fase del método simplex para problemas de minimización.

    Los problemas deben estar en forma estándar. Es decir

        mín. c^Tx
            s.a Ax  = b
                 x >= 0

    Parameters
    ----------
    A: numpy array (m x n)
        La matriz con las restricciones sobre las soluciones factibles.
    b: numpy array (m)
        El vector de restricciones sobre las soluciones factibles.
    c: numpy array (n)
        El funcional lineal a optimizar, en forma de vector.
    n_orig: int
        El número de variables artificiales.
    B: numpy array (m)
        La actuales básicas.
    N : numpy array (n-m)
        Las actuales no básicas.
    dual: boolean
        Si es True, usa el método dual.

    Returns
    --------
    r: Respuesta
        Ver documentación de clase

    """

    m = A.shape[0]
    n = A.shape[1]


    #Definir las columnas básicas y las no-básicas
    if B is None:
        B = np.arange(n-m,n)
    if N is None:
        N = np.array([i for i in range(n) if i not in B])

    conteo = 0

    while True:
        conteo = conteo + 1

        r = _simplex_one_step(A,b,c, B,N,n_orig) if not dual else _dual_simplex_one_step(A,b,c,B,N,n_orig)

        if r.termination_flag==0:
            x = np.zeros(n)
            x[B] = r.optimal_point
            return(rp.Respuesta(h=x, z0=r.optimal_value, flag=0, iter=conteo, B=r.basicas, N=r.no_basicas, n_orig=n_orig))

        elif r.termination_flag == 1 or r.termination_flag == -1:
            return(r)

        B = r.basicas
        N = r.no_basicas

def solve_min_leq(A,b,c):
    """Resuelve un problema lineal de minimización.

    Método simplex de dos fases en su forma matricial para resolver el problema

            min. c^T x
                s.a Ax <= b
                     x >= 0

    Parameters
    ----------
    A: numpy array de (m x n)
        La matriz con las restricciones sobre las soluciones factibles
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector
    dual: boolean
        True si queremos usar el método simplex dual.

    Returns
    --------
    r: Respuesta
        Ver documentación de clase.
    """
    n_orig = A.shape[1]

    A, c = utils.estandarizar(A,b,c)
    n = A.shape[1]
    m = A.shape[0]

    if len(b[b<0]) == 0:
        #Es el problema de siempre
        r = fase_ii_min(A,b,c, n_orig)
        sens = ss.analisis_de_sensibilidad_min(A,b,c,r)
        r.sensinfo = sens
        return(r)

    #Si hay alguno negativo, revisar cuáles
    first_m = np.arange(len(b))
    ind_neg = first_m[b >= 0]

    #Si uno de los b_i negativos tiene coeficientes sólo positivos en A, no hay
    #posible solución al problema
    for ind in ind_neg:
        aux = A[ind,:]
        if len(aux[aux >= 0]) == len(aux):
            return(rp.Respuesta(flag=-1))

    #Si no se da el caso de arriba, procedemos con la fase I. Primero introducimos
    #variables artificiales
    artificiales = -np.eye(m)[:,b<0]
    A_aux = np.hstack((A, artificiales))
    c_aux = np.hstack((np.zeros(n), np.ones(len(b[b<0]))))

    #Y resolvemos el problema auxiliar
    B = np.arange(len(c_aux))[c_aux == 1]
    r_aux = fase_ii_min(A_aux, b, c_aux, B=B)

    #Si el problema auxiliar no tiene solución 0, el original no tiene solución factible
    if r_aux.optimal_value != 0:
        return(rp.Respuesta(flag=-1))

    #Si el auxiliar sí tiene solución 0, hay que revisar la base que tenemos.
    #Primero revisamos si hay variables aritificiales en ella
    B = r_aux.basicas
    if max(B)<n:
        #No hay variables artificiales en la base
        return(fase_ii_min(A, b, c, B=B))

    #Si no, hay que ir sacando una por una las variables artificiales de la base.
    #Estamos suponiendo que que las restricciones del problema original eran todas
    #linealmente independientes y que no vamos a tener problemas para encontrar
    #H[i,j]>0 en al menos una variable x_j. Ver Griva, et. al. p.156
    N = r_aux.no_basicas
    A_B = np.column_stack(A[:,i] for i in B)
    A_N = np.column_stack(A[:,i] for i in N)
    H = np.dot(np.linalg.inv(A_B), A_N)
    for i in range(m):
        if B[i] < n:
            continue
        for j in range(len(N)):
            if H[i,j] > 0:
                aux = B[i]
                B[i] = N[j]
                N[j] = aux
                break
    #Ya terminamos de sacar las básicas
    return(fase_ii_min(A,b,c,B,N))

def solve_max_leq(A,b,c):
    """Resuelve un problema lineal  de maximización.

    Método simplex de dos fases en su forma matricial para resolver el problema

            máx. c^T x
                s.a Ax <= b
                    x >= 0

    Parameters
    ----------
    A: numpy array de (m x n)
        La matriz con las restricciones sobre las soluciones factibles
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector

    Returns
    --------
    r: Respuesta
        Ver documentación de clase.
    """

    respuesta = solve_min_leq(A,b,-c)
    respuesta.optimal_value = -respuesta.optimal_value

    #Corregir análisis de sensibilidad
    if respuesta.sensinfo is not None:
        new_gammas = np.zeros_like(respuesta.sensinfo.gammas)
        new_gammas[:,0] = -respuesta.sensinfo.gammas[:,1]
        new_gammas[:,1] = -respuesta.sensinfo.gammas[:,0]
        respuesta.sensinfo.gammas = new_gammas
    return(respuesta)


def simplex_dual_min_geq(A,b,c):
    """Método simplex dual.

    Resuelve problemas del tipo

        mín. c^Tx
            s.a. Ax >= b
                  x >= 0
                  c >= 0

    Aprovechando que conocemos una solución dual-factible de
    inmediato por lo probado en el documento adjunto y que entonces podemos
    proceder a la fase 2 del método dual.


    Parameters
    ----------
    A: numpy array (m x n)
        La matriz con las restricciones sobre las soluciones factibles
    b: numpy array (m x n)
        El vector de restricciones sobre las soluciones factibles
    c : numpy array (n x 1)
        El funcional lineal a optimizar, en forma de vector

    Returns
    -------
    r: Respuesta
    """

    n_orig = A.shape[1]

    A, c = utils.estandarizar(A,b,c,False)
    r = fase_ii_min(A,b,c,n_orig, dual=True)

    return(r)
