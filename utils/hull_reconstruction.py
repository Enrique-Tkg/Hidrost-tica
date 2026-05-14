
import numpy as np
import pandas as pd

from utils.splines import compute_natural_spline, interpolate


def generate_hull_surface(csv_file: str, output_grid: np.ndarray) -> np.ndarray:
    """Reconstrói a superfície do casco a partir de uma tabela de offsets CSV.

    Args:
        csv_file: Caminho para o arquivo CSV com colunas x, z, y.
        output_grid: Grade de consulta onde cada ponto é um par [x, z].
            Aceita:
                - array shape (N, 2) para N pontos
                - array shape (M, N, 2) para grade 2D
                - array shape (2,) para um único ponto

    Returns:
        Valores y interpolados no mesmo formato de grade (M, N) ou vetor 1D.
    """
    offset_data = _load_offset_table(csv_file)
    query_points, output_shape = _normalize_query_grid(output_grid)

    x_query = query_points[:, 0]
    z_query = query_points[:, 1]

    y_query = _reconstruct_surface(offset_data, x_query, z_query)

    return y_query.reshape(output_shape)


def _load_offset_table(csv_file: str) -> pd.DataFrame:
    """Carrega e valida o offset table no formato x,z,y."""
    df = pd.read_csv(csv_file)

    required_columns = {"x", "z", "y"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"O arquivo CSV deve conter as colunas {required_columns}. "
            f"Colunas presentes: {list(df.columns)}"
        )

    df = df.loc[:, ["x", "z", "y"]].copy()
    df = df.dropna(subset=["x", "z", "y"])
    df["x"] = df["x"].astype(float)
    df["z"] = df["z"].astype(float)
    df["y"] = df["y"].astype(float)

    df = (
        df.groupby(["x", "z"], as_index=False)["y"]
        .mean()
        .sort_values(["x", "z"], ignore_index=True)
    )

    if df.empty:
        raise ValueError("O arquivo CSV não contém nenhum ponto válido.")

    return df


def _normalize_query_grid(output_grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normaliza a grade de consulta para uma lista de pontos 2D."""
    grid = np.asarray(output_grid, dtype=float)

    if grid.ndim == 1 and grid.shape == (2,):
        return grid.reshape(1, 2), (1,)

    if grid.ndim == 2 and grid.shape[1] == 2:
        return grid, (grid.shape[0],)

    if grid.ndim == 3 and grid.shape[2] == 2:
        points = grid.reshape(-1, 2)
        return points, (grid.shape[0], grid.shape[1])

    raise ValueError(
        "output_grid deve ser um array com shape (N, 2), (M, N, 2) ou (2,)"
    )


def _reconstruct_surface(df: pd.DataFrame,
                         x_query: np.ndarray,
                         z_query: np.ndarray) -> np.ndarray:
    """Reconstrói valores y em pontos arbitrários x/z."""
    station_splines = _build_station_splines(df)
    query_points_by_z = _group_query_points_by_z(z_query)
    y_result = np.full(x_query.shape, np.nan, dtype=float)

    for z_value, indices in query_points_by_z.items():
        longitudinal_x, longitudinal_y = _evaluate_along_stations(station_splines, z_value)
        if len(longitudinal_x) < 2:
            raise ValueError(
                f"Não há estações suficientes para interpolar a altura longitudinal em z={z_value}. "
                "Verifique se a tabela de offsets cobre essa altura."
            )

        longitudinal_spline = compute_natural_spline(longitudinal_x, longitudinal_y)
        x_targets = x_query[indices]

        out_of_range = (x_targets < longitudinal_x[0]) | (x_targets > longitudinal_x[-1])
        if np.any(out_of_range):
            # Valores fora do domínio longitudinal não podem ser interpolados com segurança.
            y_result[indices[out_of_range]] = np.nan

        valid_indices = indices[~out_of_range]
        if valid_indices.size > 0:
            y_result[valid_indices] = interpolate(
                x_query[valid_indices],
                longitudinal_x,
                longitudinal_y,
                longitudinal_spline,
            )

    return y_result


def _build_station_splines(df: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    """Cria uma spline vertical z->y para cada estação x."""
    station_splines = []
    grouped = df.groupby("x", sort=True)

    for x_value, group in grouped:
        group_sorted = group.sort_values("z")
        z_values = group_sorted["z"].to_numpy(dtype=float)
        y_values = group_sorted["y"].to_numpy(dtype=float)

        if len(z_values) < 2:
            continue

        spline_coeffs = compute_natural_spline(z_values, y_values)
        station_splines.append(
            {
                "x": float(x_value),
                "z": z_values,
                "y": y_values,
                "spline": spline_coeffs,
            }
        )

    if not station_splines:
        raise ValueError("Não existem estações válidas no arquivo de offsets.")

    return station_splines


def _group_query_points_by_z(z_query: np.ndarray) -> dict[float, np.ndarray]:
    """Agrupa índices de consulta por valor de z para reconstrução por fatia."""
    unique_z, inverse_idx = np.unique(z_query, return_inverse=True)
    grouped = {}
    for group_idx, z_value in enumerate(unique_z):
        grouped[float(z_value)] = np.where(inverse_idx == group_idx)[0]
    return grouped


def _evaluate_along_stations(station_splines: list[dict[str, np.ndarray]],
                             z_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Avalia todas as estações no mesmo nível de z para interpolação longitudinal."""
    longitudinal_x = []
    longitudinal_y = []

    for station in station_splines:
        z_values = station["z"]
        if z_value < z_values[0] or z_value > z_values[-1]:
            continue

        y_value = interpolate(z_value, z_values, station["y"], station["spline"])
        longitudinal_x.append(station["x"])
        longitudinal_y.append(y_value)

    if not longitudinal_x:
        return np.array([], dtype=float), np.array([], dtype=float)

    sorted_indices = np.argsort(longitudinal_x)
    return (
        np.asarray(longitudinal_x, dtype=float)[sorted_indices],
        np.asarray(longitudinal_y, dtype=float)[sorted_indices],
    )
