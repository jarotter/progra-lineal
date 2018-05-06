import numpy as np
import scipy as sp
import scipy.linalg

def simplex_one_step_bland(A,b,c,B,N):
    """

    Actualiza las variables básicas (B) y no-básicas (N) de la solución
    actual al problema.

    Parámetros
    ----------
    A: matriz
        La matriz con las restricciones sobre las soluciones factibles
    b: vector
        El vector de restricciones sobre las soluciones factibles
    c : vector
        El funcional lineal a optimizar, en forma de vector
    B: vector
        Los índices de las variables básicas
    N: vector
        Los índices de las variables no básicas

    Regresa
    -------
    Una de tres posibles listas según el caso:

    1. Si en este paso se llegó a la solución, la lista contiene
        i. "Solución", indicando que terminamos
        ii. El resultado de :math: A_Bx=b
        iii. El resultado de :math: \lambda b
        iv. B
        v. N
    2. Si el problema se detectó como no acotado
        i. "No acotado", indicando que no es acotado
        ii. None
    3. Si ninguna de las anteriores pasó
        i. "Continúa", indicando que faltan iteraciones
        ii. B
        iii. N


    """

    c_B = np.array([c[i] for i in B])
    c_N = np.array([c[i] for i in N])

    #Usamos column_stack porque numpy lo hace transpuesto
    A_N = np.column_stack([A[:,i] for i in N])
    A_B = np.column_stack([A[:,i] for i in B])

    lambda_ = sp.linalg.solve(np.transpose(A_B), c_B)
    r_N = np.dot(lambda_, A_N) - c_N


    if max(r_N) <= 0:
        return ("Solución", sp.linalg.solve(A_B,b), np.dot(lambda_,b), B, N)

    entrada=-1
    #Usamos la regla de Bland
    for i in range(len(r_N)):
        if r_N[i] > 0:
            entrada = N[i]
            break


    h = sp.linalg.solve(A_B,b)
    H_e = sp.linalg.solve(A_B, A[:,entrada])

    if(max(H_e)<=0):
        return ("No acotado", None)

    salida = -1
    cociente = np.Inf
    for i in range(len(h)):
        if H_e[i]>0 and h[i]/H_e[i] < cociente:
            cociente = h[i]/H_e[i]
            salida = B[i]

    for i in range(len(B)):
        if B[i] == salida:
            B[i] = entrada

    for i in range(len(N)):
        if N[i] == entrada:
            N[i] = salida

    return ("Continúa", B, N)

def fase_ii(A,b,c, n):
    """
    Fase II del método simplex en su forma matricial para resolver el problema
    en forma estándar

        .. math::

            min. c^tx
                s.a Ax = b
                    x >= 0

    Parámetros
    ----------
    A: matriz
        La matriz con las restricciones sobre las soluciones factibles
    b: vector
        El vector de restricciones sobre las soluciones factibles
    c : vector
        El funcional lineal a optimizar, en forma de vector

    Regresa
    --------
    Un diccionario que incluye:

    SBF: Los valores de las variables básicas en la solución

    Valor óptimo: el valor de :math:`f(x) = c^T x` que se alcanza en la solución

    Estatus: Indica cómo terminó el problema. Sus valores se interpretan como
        1. (0) si se llegó a un óptimo
        2. (1) si el problema es no-acotado

    Número de iteraciones: El número de pasos que se dieron para solucionar
    el problema dado.

    """
    #Agregar variables de holgura
    m = len(b)
    A = np.concatenate((A,np.eye(m)), axis=1)
    c = np.hstack((c,np.zeros(m)))

    #Definir las columnas básicas y las no-básicas
    N = np.arange(0,A.shape[1]-m)
    B = np.arange(A.shape[1]-m,A.shape[1])


    conteo = 0

    while True:
        conteo = conteo + 1
        respuesta = simplex_one_step_bland(A,b,c,B,N)

        if respuesta[0]=="Solución":
            x = np.zeros(n)
            for i in range(len(B)):
                if B[i] < n:
                    x[B[i]] = respuesta[1][i]

            return({'SBF': x,
                    'Valor Óptimo': respuesta[2],
                    'Estatus': 0,
                    'Número de iteraciones': conteo,
                    'Base': B })

        elif respuesta[0] == "No acotado":
            return ({'Variable básica': None,
                    'Valor Óptimo': None,
                    'Estatus': 1,
                    'Número de iteraciones': conteo,
                    'Base': B})

        B = respuesta[1]
        N = respuesta[2]
