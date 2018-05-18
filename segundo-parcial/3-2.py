 # -*- coding: utf-8 -*-
import numpy as np
from src import simplex_upto_p2 as simplex
from src import printer

#Actividad 3
A = np.array([[6, 8, 3], [4, 4, 3]])
b = np.array([300,200])
c = np.array([40, 40, 20])
rd = simplex.simplex_dual_min_geq(A,b,c)
print(printer.print_resp(rd))
