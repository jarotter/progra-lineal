import numpy as np
from scipy import linalg
from scipy import sparse

def recorte(v, dv):
    """Encuentra el valor máximo 0 < alfa <= 1 tal que v+alfa*dv >=0

    Parámetros:
    ---------
    v: numpy array
        Vector v
    dv: numpy array

    Regresa
    -------
    alfa: double
    """
    try:
        cota = np.min([-(v[i])/dv[i] for i in range(len(v)) if dv[i]<0])
    except ValueError:
        cota = np.inf
    return np.min([cota,1])

def punto_primer_orden(A, b, c):
    """ Encuentra la aproximación primal-dual de orden 1.

    Por mínimos cuadrados, encuentra un punto inicial factible (si es necesario,
    lo modifica para ser factible) y "bueno" para el problema primal-dual.

    Parameters:
    -----------
    A: numpy array
        La matriz de restricciones
    b: numpy array
        El lado derecho de las restricciones
    c: numpy array
        La función a optimizar

    Returns:
    --------
    x: numpy array
        La solución factible de mínimos cuadrados del primal.
    y: numpy array
        La solución factible de mínimos cuadrados del dual.
    z: numpy array
        La solución factible de mínimos cuadrados.

    """

    is_sparse = sparse.issparse(A)
    n = len(c)
    m = len(b)

    if is_sparse:
        x = sparse.linalg.lsqr(A,b)
        x = x[0]
        y = sparse.linalg.lsqr(A.T,c)
        y = y[0]
        z = c - sparse.csc_matrix.dot(A.T,y)
    else:
        x = linalg.lstsq(A,b)
        x = x[0]
        y = linalg.lstsq(A.T,c)
        y = y[0]
        z = c - np.dot(A.T,y)

    min_x = np.min(x)
    min_z = np.min(z)

    x = x + np.max([0, -3/2*min_x])*np.ones(n)
    z = z + np.max([0, -3/2*min_z])*np.ones(n)

    if min_x == 0:
        mu = 1/2*np.dot(x,z)
        x = x + mu/np.dot(np.ones(m),z)*np.ones(n)
        z = z + mu/np.dot(np.ones(n),x)*np.ones(m)

    return(x,y,z)


def newton_pi(A,b,c, tol=1e-9):
    """ Método de Newton con trayectoria central.

    Para parejas primal-dual de la forma

    (P): min c^Tx                   (D): max b^Ty
            s.a Ax=b                       s.a. A^Ty + z = c
                x>=0                                z>=0

    Parameters:
    -----------
    A: numpy array
        La matriz de restricciones
    b: numpy array
        El lado derecho de las restricciones
    c: numpy array
        La función a optimizar

    Returns:
    --------
    x0: numpy array
        La solución óptima del problema primal.
    (y0, z0):(numpy array, numpy array):
        La solución óptima del problema dual.
    num_iter: int
        El número de iteraciones que tomó resolver el problema.

    """

    def F(x, y, z):
        """Condiciones de Karush-Kuhn Tucker para problemas lineales.

        1. Ax=b (x primal factible)
        2. A^Ty+z=c (y dual factible)
        3. xjzj = 0 para j = 1, ... n (Holgura complementaria)
        """

        Ax = sparse.csc_matrix.dot(A,x) if is_sparse else np.dot(A,x)
        ATy = sparse.csc_matrix.dot(A.T,y) if is_sparse else np.dot(A.T,y)
        return(np.concatenate([Ax-b, ATy+z-c, x*z]))

    def calcular_delta(A,x,z):
        """Calcula la dirección de descenso de Newton.
        """

        fila3 = np.hstack((np.diag(z), np.zeros((n,m)), np.diag(x)))
        if is_sparse:
            fila1 = sparse.hstack((A, np.zeros((m,m)), np.zeros((m,n))))
            fila2 = sparse.hstack((np.zeros((n,n)), A.T, np.eye(n)))
            KKT_matrix = sparse.vstack((fila1, fila2, fila3))
            KKT_matrix = sparse.csc_matrix(KKT_matrix)

        else:
            fila1 = np.hstack((A, np.zeros((m,m)), np.zeros((m,n))))
            fila2 = np.hstack((np.zeros((n,n)), A.T, np.eye(n)))
            KKT_matrix = np.vstack((fila1, fila2, fila3))

        KKT_vector = -Fxyz + np.hstack((np.zeros(n),
                                        np.zeros(m),
                                        sigma*mu*np.ones(n)))
        if is_sparse:
            delta = sparse.linalg.spsolve(KKT_matrix, KKT_vector)
        else:
            delta = linalg.solve(KKT_matrix, KKT_vector)

        return(delta)


    is_sparse = sparse.issparse(A)
    n = len(c)
    m = len(b)

    x, y, z = punto_primer_orden(A,b,c)

    k = 0
    sigma = 0.1
    const_ajuste = 999/1000

    Fxyz = F(x,y,z)
    norm = linalg.norm(Fxyz, np.inf)

    while norm > tol and k<=200:
        print('Al inicio de la iteración {nit} la norma de F es {nr}'.format(nit=k+1, nr=norm))

        mu = 1/n*np.dot(x,z)
        delta = calcular_delta(A,x,z)

        alfa_x = recorte(x, delta[0:n])
        alfa_z = recorte(z, delta[-n:])

        x = x + const_ajuste*alfa_x*delta[0:n]
        y = y + const_ajuste*alfa_z*delta[n:(n+m)]
        z = z + const_ajuste*alfa_z*delta[-n:]

        if alfa_z*alfa_x > 0.8:
            sigma = np.max([sigma/10, 10e-4])
        k = k+1
        Fxyz = F(x,y,z)
        norm = linalg.norm(Fxyz, np.inf)

    print('===========CONVERGENCIA===========')
    return(x, (y,z), k)
