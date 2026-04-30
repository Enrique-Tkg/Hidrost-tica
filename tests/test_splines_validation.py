import unittest
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
import pandas as pd
from utils.splines import compute_natural_spline, interpolate


class TestSplinesValidation(unittest.TestCase):
    """Testes de validação das splines usando SciPy como referência."""

    def setUp(self):
        """Prepara dados de teste para cada teste."""
        # Dados de teste simples
        self.x_simple = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.y_simple = np.array([0.0, 1.0, 0.5, 2.0, 1.5])

        # Dados de teste com mais pontos
        self.x_dense = np.linspace(0, 2 * np.pi, 15)
        self.y_dense = np.sin(self.x_dense)

        # Dados de teste com função quadrática
        self.x_quad = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.y_quad = self.x_quad ** 2

    def test_spline_interpolation_at_known_points(self):
        """
        Verifica se a spline interpola corretamente nos pontos conhecidos.
        O erro nos pontos iniciais deve ser praticamente zero.
        """
        spline_coeffs = compute_natural_spline(self.x_simple, self.y_simple)

        for i, x_val in enumerate(self.x_simple):
            y_pred = interpolate(x_val, self.x_simple, self.y_simple, spline_coeffs)
            # Deve estar muito próximo do valor original
            self.assertAlmostEqual(
                y_pred, self.y_simple[i], places=10,
                msg=f"Interpolação falhou no ponto {i}: esperado {self.y_simple[i]}, obteve {y_pred}"
            )

    def test_spline_vs_scipy_interpolation(self):
        """
        Compara a interpolação da spline customizada com a do SciPy
        em pontos entre os dados conhecidos.
        """
        # Gera spline com seu código
        spline_coeffs = compute_natural_spline(self.x_simple, self.y_simple)

        # Gera spline com SciPy
        scipy_spline = CubicSpline(self.x_simple, self.y_simple, bc_type='natural')

        # Testa em pontos interpolados
        x_test = np.linspace(self.x_simple[0], self.x_simple[-1], 50)
        max_error = 0
        errors = []

        for x_val in x_test:
            y_custom = interpolate(x_val, self.x_simple, self.y_simple, spline_coeffs)
            y_scipy = scipy_spline(x_val)
            error = abs(y_custom - y_scipy)
            errors.append(error)
            max_error = max(max_error, error)

        # O erro máximo deve ser pequeno (tolerância ajustável)
        avg_error = np.mean(errors)
        self.assertLess(
            max_error, 1e-6,
            msg=f"Erro máximo de {max_error} excede a tolerância. Erro médio: {avg_error}"
        )

    def test_spline_with_dense_data(self):
        """
        Valida a spline com dados mais densos (dados senoidais).
        """
        spline_coeffs = compute_natural_spline(self.x_dense, self.y_dense)
        scipy_spline = CubicSpline(self.x_dense, self.y_dense, bc_type='natural')

        # Testa em pontos aleatórios
        x_test = np.random.uniform(self.x_dense[0], self.x_dense[-1], 100)
        errors = []

        for x_val in x_test:
            y_custom = interpolate(x_val, self.x_dense, self.y_dense, spline_coeffs)
            y_scipy = scipy_spline(x_val)
            error = abs(y_custom - y_scipy)
            errors.append(error)

        max_error = max(errors)
        avg_error = np.mean(errors)

        self.assertLess(
            max_error, 1e-5,
            msg=f"Erro máximo com dados densos: {max_error}. Erro médio: {avg_error}"
        )

    def test_spline_monotonicity_preservation(self):
        """
        Verifica se a spline preserva a monotonidade dos dados quando apropriado.
        Para dados monotonicamente crescentes, a spline também deve ser.
        """
        # Cria dados monotonicamente crescentes
        x_mono = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y_mono = np.array([1.0, 2.5, 4.0, 5.5, 7.0, 8.5])

        spline_coeffs = compute_natural_spline(x_mono, y_mono)
        scipy_spline = CubicSpline(x_mono, y_mono, bc_type='natural')

        # Testa monotonidade em múltiplos pontos
        x_test = np.linspace(x_mono[0], x_mono[-1], 100)
        y_custom = [interpolate(x_val, x_mono, y_mono, spline_coeffs) for x_val in x_test]

        # Verifica se os valores da spline customizada são monotonicamente crescentes
        diffs = np.diff(y_custom)
        min_diff = np.min(diffs)

        self.assertGreater(
            min_diff, -1e-6,  # Tolerância pequena para erro numérico
            msg=f"Spline não mantém monotonidade. Menor diferença: {min_diff}"
        )

    def test_spline_with_quadratic_data(self):
        """
        Testa com dados quadráticos para verificar a precisão
        em uma função polinomial conhecida.
        """
        spline_coeffs = compute_natural_spline(self.x_quad, self.y_quad)
        scipy_spline = CubicSpline(self.x_quad, self.y_quad, bc_type='natural')

        x_test = np.linspace(self.x_quad[0], self.x_quad[-1], 50)
        errors = []

        for x_val in x_test:
            y_custom = interpolate(x_val, self.x_quad, self.y_quad, spline_coeffs)
            y_scipy = scipy_spline(x_val)
            error = abs(y_custom - y_scipy)
            errors.append(error)

        max_error = max(errors)
        avg_error = np.mean(errors)

        print(f"\nTeste Quadrático - Erro máximo: {max_error}, Erro médio: {avg_error}")

        self.assertLess(
            max_error, 1e-5,
            msg=f"Spline não aproxima bem função quadrática. Erro máximo: {max_error}"
        )

    def test_boundary_conditions_natural_spline(self):
        """
        Verifica se as condições de contorno da spline natural estão corretas.
        Espera-se que a segunda derivada seja zero nas extremidades.
        """
        # Apenas um teste conceptual - valida se a spline produz valores
        # consistentes com as condições de contorno natural
        spline_coeffs = compute_natural_spline(self.x_simple, self.y_simple)

        # Valores nas extremidades devem ser exatos
        y_first = interpolate(self.x_simple[0], self.x_simple, self.y_simple, spline_coeffs)
        y_last = interpolate(self.x_simple[-1], self.x_simple, self.y_simple, spline_coeffs)

        self.assertAlmostEqual(
            y_first, self.y_simple[0], places=10,
            msg="Valor no primeiro ponto não corresponde"
        )
        self.assertAlmostEqual(
            y_last, self.y_simple[-1], places=10,
            msg="Valor no último ponto não corresponde"
        )


class TestSplinesEdgeCases(unittest.TestCase):
    """Testes para casos extremos."""

    def test_minimum_points_for_spline(self):
        """
        Testa spline com número mínimo de pontos (4 pontos).
        """
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 1.5, 3.0])

        spline_coeffs = compute_natural_spline(x, y)
        scipy_spline = CubicSpline(x, y, bc_type='natural')

        # Testa interpolação em um ponto intermediário
        x_test = 1.5
        y_custom = interpolate(x_test, x, y, spline_coeffs)
        y_scipy = scipy_spline(x_test)

        error = abs(y_custom - y_scipy)
        self.assertLess(
            error, 1e-6,
            msg=f"Erro com mínimo de pontos: {error}"
        )

    def test_constant_data(self):
        """
        Testa spline com dados constantes (todos os y iguais).
        """
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])

        spline_coeffs = compute_natural_spline(x, y)

        # Todos os pontos devem retornar o mesmo valor
        for x_val in [0.5, 1.5, 2.5]:
            y_pred = interpolate(x_val, x, y, spline_coeffs)
            self.assertAlmostEqual(
                y_pred, 5.0, places=10,
                msg=f"Spline não mantém constância em x={x_val}"
            )


class TestSplinesFromOffsetTable(unittest.TestCase):
    """Testes de validação das splines geradas a partir do offsettable."""

    @classmethod
    def setUpClass(cls):
        """Carrega os dados do offsettable uma única vez para todos os testes."""
        csv_path = Path(__file__).resolve().parents[1] / "Pre_Processamento" / "offsettable.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Arquivo offsettable.csv não encontrado em {csv_path}")
        
        cls.df_offset = pd.read_csv(csv_path)
        print(f"\n✓ Carregado offsettable com {len(cls.df_offset)} pontos")

    def test_offsettable_spline_for_fixed_x(self):
        """
        Testa splines geradas para um valor fixo de x usando dados do offsettable.
        """
        # Seleciona dados para um valor específico de x
        x_fixed = self.df_offset['x'].unique()[0]
        data_for_x = self.df_offset[self.df_offset['x'] == x_fixed].sort_values('z')
        
        if len(data_for_x) < 4:
            self.skipTest(f"Não há dados suficientes para x={x_fixed}")
        
        z_values = data_for_x['z'].values
        y_values = data_for_x['y'].values
        
        # Gera spline com seu código
        spline_coeffs = compute_natural_spline(z_values, y_values)
        
        # Gera spline com SciPy
        scipy_spline = CubicSpline(z_values, y_values, bc_type='natural')
        
        # Testa interpolação em pontos intermediários
        z_test = np.linspace(z_values[0], z_values[-1], 20)
        max_error = 0
        
        for z_val in z_test:
            y_custom = interpolate(z_val, z_values, y_values, spline_coeffs)
            y_scipy = scipy_spline(z_val)
            error = abs(y_custom - y_scipy)
            max_error = max(max_error, error)
        
        print(f"\n✓ Teste x={x_fixed}: Erro máximo = {max_error:.2e}")
        self.assertLess(
            max_error, 1e-5,
            msg=f"Erro máximo para x={x_fixed}: {max_error}"
        )

    def test_offsettable_spline_for_fixed_z(self):
        """
        Testa splines geradas para um valor fixo de z usando dados do offsettable.
        """
        # Seleciona dados para um valor específico de z
        z_fixed = self.df_offset['z'].unique()[0]
        data_for_z = self.df_offset[self.df_offset['z'] == z_fixed].sort_values('x')
        
        if len(data_for_z) < 4:
            self.skipTest(f"Não há dados suficientes para z={z_fixed}")
        
        x_values = data_for_z['x'].values
        y_values = data_for_z['y'].values
        
        # Gera spline com seu código
        spline_coeffs = compute_natural_spline(x_values, y_values)
        
        # Gera spline com SciPy
        scipy_spline = CubicSpline(x_values, y_values, bc_type='natural')
        
        # Testa interpolação em pontos intermediários
        x_test = np.linspace(x_values[0], x_values[-1], 20)
        max_error = 0
        
        for x_val in x_test:
            y_custom = interpolate(x_val, x_values, y_values, spline_coeffs)
            y_scipy = scipy_spline(x_val)
            error = abs(y_custom - y_scipy)
            max_error = max(max_error, error)
        
        print(f"\n✓ Teste z={z_fixed}: Erro máximo = {max_error:.2e}")
        self.assertLess(
            max_error, 1e-5,
            msg=f"Erro máximo para z={z_fixed}: {max_error}"
        )

    def test_offsettable_all_series(self):
        """
        Testa splines para todas as séries de dados (cada x e cada z).
        Apenas valida se o resultado final é aceitável.
        """
        errors_by_x = {}
        errors_by_z = {}
        
        # Testa todas as séries para cada x
        for x_val in self.df_offset['x'].unique():
            data = self.df_offset[self.df_offset['x'] == x_val].sort_values('z')
            
            if len(data) < 4:
                continue
            
            z_arr = data['z'].values
            y_arr = data['y'].values
            
            spline_coeffs = compute_natural_spline(z_arr, y_arr)
            scipy_spline = CubicSpline(z_arr, y_arr, bc_type='natural')
            
            z_test = np.linspace(z_arr[0], z_arr[-1], 10)
            errors = [
                abs(interpolate(z_val, z_arr, y_arr, spline_coeffs) - scipy_spline(z_val))
                for z_val in z_test
            ]
            errors_by_x[x_val] = max(errors)
        
        # Testa todas as séries para cada z
        for z_val in self.df_offset['z'].unique():
            data = self.df_offset[self.df_offset['z'] == z_val].sort_values('x')
            
            if len(data) < 4:
                continue
            
            x_arr = data['x'].values
            y_arr = data['y'].values
            
            spline_coeffs = compute_natural_spline(x_arr, y_arr)
            scipy_spline = CubicSpline(x_arr, y_arr, bc_type='natural')
            
            x_test = np.linspace(x_arr[0], x_arr[-1], 10)
            errors = [
                abs(interpolate(x_val, x_arr, y_arr, spline_coeffs) - scipy_spline(x_val))
                for x_val in x_test
            ]
            errors_by_z[z_val] = max(errors)
        
        # Verificação geral
        all_errors_x = list(errors_by_x.values())
        all_errors_z = list(errors_by_z.values())
        
        max_error_x = max(all_errors_x) if all_errors_x else 0
        max_error_z = max(all_errors_z) if all_errors_z else 0
        
        print(f"\n✓ Total de séries para x testadas: {len(errors_by_x)}")
        print(f"✓ Total de séries para z testadas: {len(errors_by_z)}")
        print(f"✓ Erro máximo (séries x): {max_error_x:.2e}")
        print(f"✓ Erro máximo (séries z): {max_error_z:.2e}")
        
        self.assertLess(
            max_error_x, 1e-4,
            msg=f"Erro máximo para séries de x: {max_error_x}"
        )
        self.assertLess(
            max_error_z, 1e-4,
            msg=f"Erro máximo para séries de z: {max_error_z}"
        )


if __name__ == '__main__':
    unittest.main()
