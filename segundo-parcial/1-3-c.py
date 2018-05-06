 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex

#Actividad 1: 3.c)
print('Actividad 1: 3.c')
A = np.array([[6,4], [8,4], [3,3]])
b = np.array([40, 40, 20])
c = np.array([375, 200])
r = simplex.solve_max_leq(A,b,c)
print(r)
c = np.array([375, 175])
r = simplex.solve_max_leq(A,b,c)
print(r)
print()
