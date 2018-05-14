import numpy as np
import newton
from scipy import linalg

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

    A = np.hstack((A, np.eye(m)))
    c = np.hstack((c, np.zeros(m)))
    return(A,b,c)

for m in [10,12,14,16]:
    A,b,c = genera_klee_minty(m)
    x0, (y0,z0), num_iter = newton.newton_pi(A, b, c, is_sparse=False)
    x0 = x0[:-m]
    c = c[:-m]
    z0 = z0[:-m]
    print(f'Klee-Minty para m = {m}')
    print(f'El valor óptimo es {np.dot(c,x0)}')
    print(f'||diag(x)z||_\inf = {linalg.norm(x0*z0)}')
    print(f'Número de iteraciones: {num_iter}')
    print(f'La solución óptima primal es: {x0}')
    print('Que en la última coordenada es el valor óptimo, como debería.')
    print(f'la solución del dual es {y0}')
    print(f'Que está bien, puesto que, multiplicando tenemos {np.dot(b,y0)}')
    print()
