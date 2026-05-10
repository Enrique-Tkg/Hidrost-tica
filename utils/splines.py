import numpy as np


def compute_natural_spline(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Calcula os coeficientes da spline cúbica natural.
    
    Uma spline cúbica natural é composta por polinômios cúbicos contínuos
    com segunda derivada zero nos pontos finais.
    
    Args:
        x: Coordenadas x dos pontos (deve estar ordenado)
        y: Coordenadas y dos pontos
        
    Returns:
        Dicionário com coeficientes: a, b, c, d (para cada segmento)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    num_points = len(x)
    
    if num_points < 2:
        raise ValueError("Pelo menos 2 pontos são necessários")
    
    if num_points == 2:
        interval_size = x[1] - x[0]
        slope = (y[1] - y[0]) / interval_size
        return {
            'a': y,
            'b': np.array([slope]),
            'c': np.zeros(num_points),
            'd': np.zeros(num_points - 1),
            'x': x,
            'h': np.array([interval_size])
        }
    
    interval_sizes = np.diff(x)
    y_differences = np.diff(y)
    
    if num_points == 3:
        second_derivatives = np.zeros(num_points)
        second_derivatives[1] = 6 * (y_differences[1] / interval_sizes[1] - 
                                      y_differences[0] / interval_sizes[0]) / \
                               (2 * (interval_sizes[0] + interval_sizes[1]))
    else:
        second_derivatives = _solve_for_second_derivatives(interval_sizes, y_differences, num_points)
    
    coefficients_a = y.copy()
    coefficients_d = np.zeros(num_points - 1)
    coefficients_b = np.zeros(num_points - 1)
    coefficients_c = np.zeros(num_points - 1)
    
    for segment_idx in range(num_points - 1):
        coefficients_c[segment_idx] = second_derivatives[segment_idx] / 2
        coefficients_d[segment_idx] = (second_derivatives[segment_idx + 1] - 
                                       second_derivatives[segment_idx]) / (6 * interval_sizes[segment_idx])
        coefficients_b[segment_idx] = (y_differences[segment_idx] / interval_sizes[segment_idx] - 
                                       interval_sizes[segment_idx] * 
                                       (2 * second_derivatives[segment_idx] + 
                                        second_derivatives[segment_idx + 1]) / 6)
    
    return {
        'a': coefficients_a,
        'b': coefficients_b,
        'c': coefficients_c,
        'd': coefficients_d,
        'x': x,
        'h': interval_sizes
    }


def _solve_for_second_derivatives(interval_sizes: np.ndarray, 
                                   y_differences: np.ndarray, 
                                   num_points: int) -> np.ndarray:
    """
    Resolve o sistema tridiagonal para encontrar as segundas derivadas.
    
    As segundas derivadas são calculadas nos pontos de controle, resolvendo
    um sistema linear tridiagonal que garante continuidade e suavidade da spline.
    
    Args:
        interval_sizes: Tamanho de cada intervalo entre pontos consecutivos
        y_differences: Diferença de y entre pontos consecutivos
        num_points: Número total de pontos de controle
        
    Returns:
        Array com as segundas derivadas em cada ponto
    """
    num_interior_points = num_points - 2
    
    main_diagonal = np.zeros(num_interior_points)
    upper_diagonal = np.zeros(num_interior_points - 1)
    lower_diagonal = np.zeros(num_interior_points - 1)
    rhs_vector = np.zeros(num_interior_points)
    
    for point_idx in range(num_interior_points):
        main_diagonal[point_idx] = 2 * (interval_sizes[point_idx] + 
                                        interval_sizes[point_idx + 1])
        rhs_vector[point_idx] = 6 * (y_differences[point_idx + 1] / 
                                     interval_sizes[point_idx + 1] - 
                                     y_differences[point_idx] / 
                                     interval_sizes[point_idx])
        
        if point_idx < num_interior_points - 1:
            upper_diagonal[point_idx] = interval_sizes[point_idx + 1]
            lower_diagonal[point_idx] = interval_sizes[point_idx + 1]
    
    interior_second_derivatives = _solve_tridiagonal(lower_diagonal, main_diagonal, 
                                                      upper_diagonal, rhs_vector)
    
    all_second_derivatives = np.zeros(num_points)
    all_second_derivatives[1:-1] = interior_second_derivatives
    
    return all_second_derivatives


def _solve_tridiagonal(lower_diagonal: np.ndarray, main_diagonal: np.ndarray, 
                       upper_diagonal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Resolve sistema linear tridiagonal usando algoritmo de Thomas.
    
    O algoritmo funciona em duas fases:
    1. Forward sweep: elimina os elementos da diagonal inferior
    2. Back substitution: resolve de trás para frente
    
    Args:
        lower_diagonal: Elementos da diagonal inferior (abaixo da diagonal principal)
        main_diagonal: Elementos da diagonal principal
        upper_diagonal: Elementos da diagonal superior (acima da diagonal principal)
        rhs: Lado direito do sistema (right-hand side)
        
    Returns:
        Solução do sistema Ax = rhs
    """
    matrix_size = len(main_diagonal)
    
    if matrix_size == 1:
        return np.array([rhs[0] / main_diagonal[0]])
    
    upper_coefficients = np.zeros(matrix_size - 1)
    rhs_coefficients = np.zeros(matrix_size)
    
    upper_coefficients[0] = upper_diagonal[0] / main_diagonal[0]
    rhs_coefficients[0] = rhs[0] / main_diagonal[0]
    
    for row_idx in range(1, matrix_size):
        denominator = main_diagonal[row_idx] - lower_diagonal[row_idx - 1] * upper_coefficients[row_idx - 1]
        
        if row_idx < matrix_size - 1:
            upper_coefficients[row_idx] = upper_diagonal[row_idx] / denominator
        
        rhs_coefficients[row_idx] = (rhs[row_idx] - 
                                     lower_diagonal[row_idx - 1] * rhs_coefficients[row_idx - 1]) / denominator
    
    solution = np.zeros(matrix_size)
    solution[matrix_size - 1] = rhs_coefficients[matrix_size - 1]
    
    for row_idx in range(matrix_size - 2, -1, -1):
        solution[row_idx] = rhs_coefficients[row_idx] - upper_coefficients[row_idx] * solution[row_idx + 1]
    
    return solution


def interpolate(x_query: float, x: np.ndarray, y: np.ndarray, 
                spline_coeffs: dict) -> float:
    """
    Interpola um ou múltiplos pontos usando a spline cúbica natural.
    
    Para cada ponto de consulta:
    1. Verifica se está em um ponto de controle (retorna o valor exato)
    2. Caso contrário, encontra o segmento apropriado
    3. Avalia o polinômio cúbico: S(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    
    Args:
        x_query: Ponto(s) onde interpolar (float ou array)
        x: Coordenadas x dos pontos de controle
        y: Coordenadas y dos pontos de controle
        spline_coeffs: Dicionário com coeficientes da spline
        
    Returns:
        Valor(es) interpolado(s) no ponto(s) x_query
    """
    coefficients_a = spline_coeffs['a']
    coefficients_b = spline_coeffs['b']
    coefficients_c = spline_coeffs['c']
    coefficients_d = spline_coeffs['d']
    control_points_x = spline_coeffs['x']
    
    scalar_input = not isinstance(x_query, (np.ndarray, list))
    
    if scalar_input:
        x_query = np.array([x_query])
    else:
        x_query = np.asarray(x_query)
    
    interpolated_values = np.zeros_like(x_query, dtype=float)
    
    for query_idx, query_point in enumerate(x_query):
        tolerance = 1e-14
        
        control_point_idx = -1
        for point_idx, control_point in enumerate(control_points_x):
            if abs(query_point - control_point) < tolerance:
                control_point_idx = point_idx
                break
        
        if control_point_idx >= 0:
            interpolated_values[query_idx] = y[control_point_idx]
        else:
            segment_idx = np.searchsorted(control_points_x[1:], query_point)
            segment_idx = min(segment_idx, len(control_points_x) - 2)
            segment_idx = max(segment_idx, 0)
            
            segment_start_x = control_points_x[segment_idx]
            distance_from_start = query_point - segment_start_x
            
            cubic_value = (coefficients_a[segment_idx] + 
                          coefficients_b[segment_idx] * distance_from_start + 
                          coefficients_c[segment_idx] * distance_from_start**2 + 
                          coefficients_d[segment_idx] * distance_from_start**3)
            
            interpolated_values[query_idx] = cubic_value
    
    if scalar_input:
        return float(interpolated_values[0])
    else:
        return interpolated_values