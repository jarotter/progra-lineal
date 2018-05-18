 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex
from src import printer

#Actividad 1: 3.e)
print('Actividad 1: 3.e)')
A = np.array([[6,4], [8,4], [3,3]])

c = np.array([300, 200])
b = np.array([45, 40, 20])
r = simplex.solve_max_leq(A,b,c)
print(r)
b = np.array([40, 45, 20])
r = simplex.solve_max_leq(A,b,c)
print(r)
b = np.array([40, 40, 25])
r = simplex.solve_max_leq(A,b,c)
print(printer.print_resp(r))
