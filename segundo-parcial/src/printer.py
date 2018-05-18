def print_resp(r):
    if r.termination_flag == 2:
        return('This could be implemented for debugging purposes.')
    if r.termination_flag == -1:
        return('Conjunto factible vacío')
    if r.termination_flag == 1:
        return(f'''Problema no acotado.
                La dirección de descenso es {r.optimal_point} + x_{r.descent_variable}*{r.descent_direction}
                B = {r.basicas}
                N = {r.no_basicas}
                Terminamos en {r.iter} iteraciones''')
    return(f'''Encontramos la solución óptima en {r.iter} iteraciones.
    x* = {r.optimal_point[0:r.n_orig]}
    z0 = {r.optimal_value}
    B = {r.basicas[r.basicas < r.n_orig]}
    N = {r.no_basicas[r.no_basicas < r.n_orig]}
    Análisis de sensibilidad: {r.sensinfo if r.sensinfo is not None else 'no está implementado para este problema'}''')
