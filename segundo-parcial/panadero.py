 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex

c = np.array([24, 14])
A = np.array([[3,2], [4,1], [2,1]])
b = np.array([120, 100, 70])

r = simplex.solve_max_leq(A,b,c)
print(r)
