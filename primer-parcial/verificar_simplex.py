import numpy as np
import simplex as simplex

A = np.array([[1, 0], [0, 2], [3,2]])
c = np.array([-3, -5])
b = np.array([4, 12, 18])

print(simplex.fase_ii(A,b,c, A.shape[1]))
