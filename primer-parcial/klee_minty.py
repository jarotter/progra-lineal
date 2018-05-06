import simplex as simplex
import numpy as np
import time

def genera_klee_minty(m):
    """
    Genera problemas de Klee Minty para la m dada.

    Parámetros
    ----------
    m: entero
        El número de variables

    Regresa
    --------
    A: matriz
        La matriz de restricciones
    b: vector
        El vector de restricciones
    c: vector
        El funcional a optimizar

    """
    c = -np.ones(m)
    A = np.eye(m, dtype=float)
    b = np.array([(2**(i+1))-1 for i in range(m)],dtype=float)

    for i in range(m):
        for j in range(i):
            A[i,j]= 2

    return(A,b,c)

def simplex_klee_minty(m):
    """
    Resuelve un problema de Klee-Minty usando el método simplex.

    Parámetros
    ----------
    m: entero
        El número de variables

    Regresa
    -------
    Una lista que contiene
        1. El número de variables
        2. El número de iteraciones que tomó resolverlo
        3. El tiempo que tomó resolverlo
    """
    A, b, c = genera_klee_minty(m)
    n = A.shape[1]

    inicio = time.time()
    resultado = simplex.fase_ii(A,b,c,n)
    final = time.time()

    return(m, resultado['Número de iteraciones'], final-inicio)

#Corremos una vez para alocar memoria y que los tiempos den lo que deberían dar
simplex_klee_minty(2)

for i in range(3,11):
    print(i)
    print(simplex_klee_minty(i))
    print()
