import numpy as np
import pandas as pd

def compute_natural_spline(x: np.ndarray, y: np.ndarray) -> dict
    # Calcula todos os coeficientes da spline cúbica natural

def interpolate(x_query: float, x: np.ndarray, y: np.ndarray, spline_coeffs: dict) -> float
    # Interpola um ou múltiplos pontos usando a spline