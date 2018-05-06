 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex

#Actividad 1: 3.b), 3.d), 3.f)
print('Actividad 1: 3.b), 3.d), 3.f)')
c = np.array([300, 200])
A = np.array([[6,4], [8,4], [3,3]])
b = np.array([40, 30, 20])
r = simplex.solve_max_leq(A,b,c)
print(r)
