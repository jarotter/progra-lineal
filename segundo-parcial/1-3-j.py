 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex


#Actividad 1: 3.j)
print('Actividad 1: 3.j')
A = np.array([[6,4], [8,4], [3,3]])
c = np.array([300, 200])
b1 = np.array([40, 40, 20])
b2 = np.array([40, 40, 21])
r1 = simplex.solve_max_leq(A,b1,c).optimal_value
r2 = simplex.solve_max_leq(A,b2,c).optimal_value
print(f'El precio sombra para b_3 está dado por {r2} - {r1} = {r2-r1}')
