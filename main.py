import numpy as np
import pandas as pd
from utils.splines import compute_natural_spline, interpolate
from utils.hull_reconstruction import generate_hull_surface
import matplotlib.pyplot as plt

def main() -> None
    # Coordena: pré-processamento → splines → reconstrução → visualização