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
    print('Klee-Minty para m = {m}'.format(m=m))
    A,b,c = genera_klee_minty(m)
    x0, (y0,z0), num_iter = newton.newton_pi(A, b, c)
    x0 = x0[:-m]
    c = c[:-m]
    z0 = z0[:-m]
    print('El valor óptimo es {vop}'.format(vop=np.dot(c,x0)))
    print('||diag(x)z||_\inf = {nrm}'.format(nrm=linalg.norm(x0*z0)))
    print('Número de iteraciones: {nt}'.format(nt=num_iter))
    print('La solución óptima primal es: {sop}'.format(sop=x0))
    print('Que en la última coordenada es el valor óptimo, como debería.')
    print('la solución del dual es {dop}'.format(dop=y0))
    print('Que está bien, puesto que, multiplicando tenemos {nd}'.format(nd=np.dot(b,y0)))
    print()
