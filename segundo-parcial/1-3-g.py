 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex

#Actividad 1: 3.g)
print('Actividad 1: 3.g)')
c = np.array([300, 200])
A = np.array([[6,4], [8,4], [3,3]])
b1 = [35, 37, 39, 41, 43, 45]
b3 = [15, 17, 19, 21, 23, 25]
for x in b1 :
    b = np.array([x, 40, 20])
    r = simplex.solve_max_leq(A,b,c)
    print(r)
for x in b1 :
    b = np.array([40, x, 20])
    r = simplex.solve_max_leq(A,b,c)
    print(r)
for x in b3 :
    b = np.array([40, 40, x])
    r = simplex.solve_max_leq(A,b,c)
    print(r)
