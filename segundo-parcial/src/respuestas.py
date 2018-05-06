 # -*- coding: utf-8 -*-
class Sensibilidad:
    """Análisis de sensibilidad al usar el método Simplex.

    Está implementado únicamente para problemas del tipo

        máx. c^Tx
            s.a. Ax <= b
                  x >= 0
                  b >= 0

    Attributes
    ----------
    lambda: numpy array (m x 1)
        La solución al problema dual.
    gammas: numpy array (n x 2)
        Los intervalos sobre los cuales variar c no afecta B.
    betas: numpy array (m x 2)
        Los intervalos sobre los cuales variar b no afecta B.
    """

    def __init__(self, l, g, b, n_orig, m):
        self.lambdas = l
        self.gammas = g
        self.betas = b
        self.n_orig = n_orig
        self.valm = m

    def __str__(self):
        return(f'''
        La solución del dual es
        {self.lambdas}
        Los intervalos para el objetivo son
        {list(self.gammas[0:self.n_orig])}
        Los intervalos para las restricciones son
        {list(self.betas[0:self.valm])}''')

class Respuesta:
    """Respuesta de dar un paso de simplex con la regla de bland.

    Attributes
    ----------
    termination_flag: int
        Indica en qué terminó el paso actual. Vale
        * (-1) si el problema tiene conjunto factible vacío
        * (0) si encontramos la solución óptima
        * (1) si el problema es no-acotado
        * (2) si debemos continuar
    optimal_point: numpy array
        La solución al sistema A_Bx = b, si el problema tiene solución óptima.
    descent_variable: int
        Si el problema es no acotado, la variable que podemos aumentar infinitamente.
    descent_direction: numpy array
        Debe interpretarse junto con var_desc de la siguiente manera: h + x[var_desc]*d es
        la dirección de descenso.
    optimal_value: double
        El valor óptimo, si existe, o -Inf si el problema es no acotado.
    B: numpy array
        Variables básicas.
    N: numpy array
        Variables no-básicas.
    iter: int
        El número de iteraciones que dio Simplex.
    sensinfo: Sensibilidad:
        El análisis de sensibilidad. (Ver documentación de clase)
    """

    def __init__(self, flag, h=None,  var_desc=None, d=None, z0=None,
        B=None, N=None, sensibilidad=None, iter=-1, n_orig=None):
        self.termination_flag = flag
        self.optimal_point = h
        self.descent_variable = var_desc
        self.descent_direction = d
        self.optimal_value = z0
        self.basicas = B
        self.no_basicas = N
        self.iter = iter
        self.sensinfo = None
        self.n_orig = n_orig

    def __str__(self):
        if self.termination_flag == 2:
            return('This could be implemented for debugging purposes.')
        if self.termination_flag == -1:
            return('Conjunto factible vacío')
        if self.termination_flag == 1:
            return(f'''Problema no acotado.
                    La dirección de descenso es {self.optimal_point} + x_{self.descent_variable}*{self.descent_direction}
                    B = {self.basicas}
                    N = {self.no_basicas}
                    Terminamos en {self.iter} iteraciones''')
        return(f'''Encontramos la solución óptima en {self.iter} iteraciones.
        x* = {self.optimal_point[0:self.n_orig]}
        z0 = {self.optimal_value}
        B = {self.basicas[self.basicas < self.n_orig]}
        N = {self.no_basicas[self.no_basicas < self.n_orig]}
        Análisis de sensibilidad: {self.sensinfo if self.sensinfo is not None else 'no está implementado para este problema'}''')
