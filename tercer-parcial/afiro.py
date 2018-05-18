import numpy as np
import scipy.io as sio
from scipy import linalg
import newton

problema = sio.loadmat('test/data/afiro.mat')
A = problema['A']
b = problema['b'].flatten()
c = problema['c'].flatten()
x0, (y0,z0), num_iter = newton.newton_pi(A, b, c)
print('El valor óptimo es {vop}'.format(vop=np.dot(c,x0)))
print('||diag(x)z||_\inf = {nrm}'.format(nrm=linalg.norm(np.dot(np.diag(x0), z0))))
print('Número de iteraciones: {nt}'.format(nt=num_iter))
