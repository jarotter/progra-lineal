import numpy as np
import scipy.io as sio
from scipy import linalg
import newton

problema = sio.loadmat('test/data/afiro.mat')
A = problema['A']
b = problema['b'].flatten()
c = problema['c'].flatten()
x0, (y0,z0), num_iter = newton.newton_pi(A, b, c, is_sparse=True)
print(f'El valor óptimo es {np.dot(c,x0)}')
print(f'||diag(x)z||_\inf = {linalg.norm(np.dot(np.diag(x0), z0))}')
print(f'Número de iteraciones: {num_iter}')
