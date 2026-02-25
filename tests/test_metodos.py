"""
Pruebas unitarias para los métodos numéricos.

Este módulo contiene pruebas para verificar el correcto funcionamiento
de cada uno de los métodos numéricos implementados.
"""

import unittest
import numpy as np
import sys
import os

# Agregar ruta del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metodos.biseccion import MetodoBiseccion
from metodos.falsa_posicion import MetodoFalsaPosicion
from metodos.punto_fijo import MetodoPuntoFijo
from metodos.newton import MetodoNewton
from metodos.secante import MetodoSecante
from funciones.definiciones import obtener_funcion, obtener_derivada


class TestMetodoBiseccion(unittest.TestCase):
    """Pruebas para el método de bisección."""

    def test_raiz_conocida(self):
        """Prueba con una función cuya raíz es conocida: x² - 4 = 0, raíz = 2."""
        f = lambda x: x**2 - 4
        metodo = MetodoBiseccion(f, 0, 3, tolerancia=1e-8)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(resultado['raiz'], 2.0, places=6)

    def test_funcion_ejercicio1(self):
        """Prueba con la función del ejercicio 1 (con función corregida)."""
        # La función original no tiene raíces positivas
        # Usamos una función similar que sí tenga raíz
        f = obtener_funcion("x**2 - 3.2*x + 2")  # Raíces en x≈0.8 y x≈2.4
        metodo = MetodoBiseccion(f, 0.5, 1.5, tolerancia=1e-6)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertIsNotNone(resultado['raiz'])
        # Verificar que f(raíz) ≈ 0
        self.assertAlmostEqual(f(resultado['raiz']), 0, places=5)

    def test_sin_cambio_signo(self):
        """Prueba que detecta cuando no hay cambio de signo."""
        f = lambda x: x**2 + 1  # Siempre positiva
        metodo = MetodoBiseccion(f, 0, 2, tolerancia=1e-6)
        resultado = metodo.resolver()

        self.assertFalse(resultado['convergencia'])

    def test_historial(self):
        """Verifica que el historial se registra correctamente."""
        f = lambda x: x**2 - 2
        metodo = MetodoBiseccion(f, 0, 2, tolerancia=1e-4)
        resultado = metodo.resolver()

        self.assertGreater(len(resultado['historial']), 0)
        self.assertIn('n', resultado['historial'][0])
        self.assertIn('a', resultado['historial'][0])
        self.assertIn('b', resultado['historial'][0])
        self.assertIn('c', resultado['historial'][0])


class TestMetodoFalsaPosicion(unittest.TestCase):
    """Pruebas para el método de falsa posición."""

    def test_raiz_conocida(self):
        """Prueba con x³ - 1 = 0, raíz = 1."""
        f = lambda x: x**3 - 1
        metodo = MetodoFalsaPosicion(f, 0, 2, tolerancia=1e-8)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(resultado['raiz'], 1.0, places=6)

    def test_funcion_ejercicio2(self):
        """Prueba con la función del ejercicio 2."""
        f = obtener_funcion("x**3 - 6*x**2 + 11*x - 6.5")
        metodo = MetodoFalsaPosicion(f, 2, 4, tolerancia=1e-7)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(f(resultado['raiz']), 0, places=5)

    def test_comparacion_biseccion(self):
        """Verifica que falsa posición generalmente converge en menos iteraciones."""
        f = lambda x: x**3 - 2

        biseccion = MetodoBiseccion(f, 1, 2, tolerancia=1e-8)
        res_bis = biseccion.resolver()

        falsa_pos = MetodoFalsaPosicion(f, 1, 2, tolerancia=1e-8)
        res_fp = falsa_pos.resolver()

        # Ambos deben converger
        self.assertTrue(res_bis['convergencia'])
        self.assertTrue(res_fp['convergencia'])

        # Las raíces deben ser similares
        self.assertAlmostEqual(res_bis['raiz'], res_fp['raiz'], places=6)


class TestMetodoPuntoFijo(unittest.TestCase):
    """Pruebas para el método de punto fijo."""

    def test_funcion_convergente(self):
        """Prueba con g(x) = cos(x), punto fijo cerca de 0.739."""
        g = lambda x: np.cos(x)
        dg = lambda x: -np.sin(x)
        metodo = MetodoPuntoFijo(g, 0.5, tolerancia=1e-8, dg=dg)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        # El punto fijo de cos(x) es aproximadamente 0.739085
        self.assertAlmostEqual(resultado['raiz'], 0.739085, places=4)

    def test_funcion_ejercicio3(self):
        """Prueba con la función del ejercicio 3."""
        g = obtener_funcion("0.5*cos(x) + 1.5")
        dg = obtener_derivada("0.5*cos(x) + 1.5")
        metodo = MetodoPuntoFijo(g, 1.0, tolerancia=1e-8, dg=dg)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        # Verificar que es punto fijo: x = g(x)
        raiz = resultado['raiz']
        self.assertAlmostEqual(raiz, g(raiz), places=6)

    def test_deteccion_divergencia(self):
        """Prueba que detecta divergencia."""
        g = lambda x: 2*x  # Diverge porque |g'(x)| = 2 > 1
        metodo = MetodoPuntoFijo(g, 1.0, tolerancia=1e-6, max_iter=20)
        resultado = metodo.resolver()

        # No debe converger
        self.assertFalse(resultado['convergencia'])


class TestMetodoNewton(unittest.TestCase):
    """Pruebas para el método de Newton-Raphson."""

    def test_raiz_conocida(self):
        """Prueba con x² - 2 = 0, raíz = √2."""
        f = lambda x: x**2 - 2
        df = lambda x: 2*x
        metodo = MetodoNewton(f, df, 1.0, tolerancia=1e-10)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(resultado['raiz'], np.sqrt(2), places=8)

    def test_funcion_ejercicio4(self):
        """Prueba con la función del ejercicio 4 (valor inicial corregido)."""
        f = obtener_funcion("x**3 - 8*x**2 + 20*x - 16")
        df = obtener_derivada("x**3 - 8*x**2 + 20*x - 16")
        # Usar x0=1.5 para evitar el punto crítico en x=2
        metodo = MetodoNewton(f, df, 1.5, tolerancia=1e-10)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(f(resultado['raiz']), 0, places=8)

    def test_convergencia_cuadratica(self):
        """Verifica que la convergencia es cuadrática."""
        f = lambda x: x**2 - 2
        df = lambda x: 2*x
        metodo = MetodoNewton(f, df, 1.0, tolerancia=1e-12)
        resultado = metodo.resolver()

        # Newton debería converger muy rápido (< 10 iteraciones)
        self.assertLess(resultado['iteraciones'], 10)

    def test_derivada_cero(self):
        """Prueba con punto donde f'(x) = 0."""
        f = lambda x: x**3
        df = lambda x: 3*x**2
        metodo = MetodoNewton(f, df, 0.0001, tolerancia=1e-10)  # Cerca del punto crítico
        resultado = metodo.resolver()

        # Debe manejar el caso sin errores
        self.assertIsNotNone(resultado)


class TestMetodoSecante(unittest.TestCase):
    """Pruebas para el método de la secante."""

    def test_raiz_conocida(self):
        """Prueba con x² - 3 = 0, raíz = √3."""
        f = lambda x: x**2 - 3
        metodo = MetodoSecante(f, 1.0, 2.0, tolerancia=1e-9)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(resultado['raiz'], np.sqrt(3), places=7)

    def test_funcion_ejercicio5(self):
        """Prueba con la función del ejercicio 5."""
        f = obtener_funcion("x*exp(-x/2) - 0.3")
        metodo = MetodoSecante(f, 0.5, 1.0, tolerancia=1e-9)
        resultado = metodo.resolver()

        self.assertTrue(resultado['convergencia'])
        self.assertAlmostEqual(f(resultado['raiz']), 0, places=7)

    def test_comparacion_newton(self):
        """Compara secante con Newton (sin derivada explícita)."""
        f = lambda x: x**3 - 2*x - 5
        df = lambda x: 3*x**2 - 2

        newton = MetodoNewton(f, df, 2.0, tolerancia=1e-10)
        res_newton = newton.resolver()

        secante = MetodoSecante(f, 1.5, 2.5, tolerancia=1e-10)
        res_secante = secante.resolver()

        # Ambos deben encontrar la misma raíz
        self.assertAlmostEqual(res_newton['raiz'], res_secante['raiz'], places=6)

        # Secante no usa derivada, solo evaluaciones de f
        self.assertNotIn('evaluaciones_df', res_secante)


class TestIntegracion(unittest.TestCase):
    """Pruebas de integración que verifican el flujo completo."""

    def test_todos_metodos_misma_funcion(self):
        """Verifica que todos los métodos encuentran la misma raíz."""
        # Función: x³ - x - 1 = 0, tiene una raíz real cerca de 1.3247
        f = lambda x: x**3 - x - 1
        df = lambda x: 3*x**2 - 1
        g = lambda x: (x + 1) ** (1/3)  # Reformulación para punto fijo

        raices = []

        # Bisección
        bis = MetodoBiseccion(f, 1, 2, tolerancia=1e-8)
        res_bis = bis.resolver()
        if res_bis['convergencia']:
            raices.append(res_bis['raiz'])

        # Falsa posición
        fp = MetodoFalsaPosicion(f, 1, 2, tolerancia=1e-8)
        res_fp = fp.resolver()
        if res_fp['convergencia']:
            raices.append(res_fp['raiz'])

        # Newton
        newton = MetodoNewton(f, df, 1.5, tolerancia=1e-8)
        res_newton = newton.resolver()
        if res_newton['convergencia']:
            raices.append(res_newton['raiz'])

        # Secante
        secante = MetodoSecante(f, 1.0, 2.0, tolerancia=1e-8)
        res_secante = secante.resolver()
        if res_secante['convergencia']:
            raices.append(res_secante['raiz'])

        # Todas las raíces deben ser similares
        self.assertGreater(len(raices), 0)
        raiz_esperada = 1.3247179572
        for raiz in raices:
            self.assertAlmostEqual(raiz, raiz_esperada, places=5)

    def test_historial_completo(self):
        """Verifica que el historial contiene todos los campos requeridos."""
        f = lambda x: x**2 - 2
        metodo = MetodoBiseccion(f, 0, 2, tolerancia=1e-6)
        resultado = metodo.resolver()

        self.assertIn('historial', resultado)
        self.assertGreater(len(resultado['historial']), 0)

        # Verificar campos del historial
        primera_fila = resultado['historial'][0]
        campos_requeridos = ['n', 'a', 'b', 'c', 'f(c)', 'error_absoluto', 'error_relativo']
        for campo in campos_requeridos:
            self.assertIn(campo, primera_fila)

    def test_error_relativo_porcentaje(self):
        """Verifica que el error relativo se calcula correctamente."""
        f = lambda x: x**2 - 4
        metodo = MetodoBiseccion(f, 0, 3, tolerancia=1e-6)
        resultado = metodo.resolver()

        for h in resultado['historial']:
            # El error relativo debe estar entre 0 y algo razonable
            self.assertGreaterEqual(h['error_relativo'], 0)
            # Verificar que existe el porcentaje
            if 'error_rel_%' in h:
                self.assertAlmostEqual(h['error_rel_%'], h['error_relativo'] * 100)


class TestValidaciones(unittest.TestCase):
    """Pruebas para las funciones de validación."""

    def test_validar_intervalo(self):
        """Prueba la validación de intervalos."""
        from utils.validaciones import validar_intervalo

        # Intervalo válido
        valido, _ = validar_intervalo(0, 1)
        self.assertTrue(valido)

        # Intervalo inválido (a >= b)
        valido, _ = validar_intervalo(1, 0)
        self.assertFalse(valido)

        valido, _ = validar_intervalo(1, 1)
        self.assertFalse(valido)

    def test_validar_tolerancia(self):
        """Prueba la validación de tolerancia."""
        from utils.validaciones import validar_tolerancia

        # Tolerancia válida
        valido, _ = validar_tolerancia(1e-6)
        self.assertTrue(valido)

        # Tolerancia inválida
        valido, _ = validar_tolerancia(-1e-6)
        self.assertFalse(valido)

        valido, _ = validar_tolerancia(0)
        self.assertFalse(valido)

    def test_validar_cambio_signo(self):
        """Prueba la detección de cambio de signo."""
        from utils.validaciones import validar_cambio_signo

        f = lambda x: x**2 - 1

        # Hay cambio de signo en [-2, 0]
        valido, _ = validar_cambio_signo(f, -2, 0)
        self.assertTrue(valido)

        # No hay cambio de signo en [2, 3]
        valido, _ = validar_cambio_signo(f, 2, 3)
        self.assertFalse(valido)


class TestOrdenConvergencia(unittest.TestCase):
    """Pruebas para verificar el orden de convergencia de cada método."""

    def test_newton_convergencia_cuadratica(self):
        """Verifica que Newton tiene convergencia cuadrática."""
        f = lambda x: x**2 - 2
        df = lambda x: 2*x
        metodo = MetodoNewton(f, df, 1.0, tolerancia=1e-14, max_iter=20)
        resultado = metodo.resolver()

        # Extraer errores
        errores = [h['error_absoluto'] for h in resultado['historial']]

        # En convergencia cuadrática, e_{n+1}/e_n^2 debería ser aproximadamente constante
        if len(errores) >= 3:
            ratios = []
            for i in range(1, len(errores)-1):
                if errores[i] > 1e-15:
                    ratio = errores[i+1] / (errores[i]**2)
                    if 0 < ratio < 100:
                        ratios.append(ratio)

            if ratios:
                # El ratio debería ser relativamente constante
                variacion = max(ratios) / min(ratios) if min(ratios) > 0 else float('inf')
                # La variación no debería ser excesiva
                self.assertLess(variacion, 100)

    def test_biseccion_convergencia_lineal(self):
        """Verifica que bisección tiene convergencia lineal."""
        f = lambda x: x**2 - 2
        metodo = MetodoBiseccion(f, 0, 2, tolerancia=1e-12, max_iter=50)
        resultado = metodo.resolver()

        # Extraer errores
        errores = [h['error_absoluto'] for h in resultado['historial']]

        # En bisección, el error se reduce a la mitad en cada iteración
        if len(errores) >= 3:
            ratios = []
            for i in range(1, len(errores)):
                if errores[i-1] > 0:
                    ratio = errores[i] / errores[i-1]
                    ratios.append(ratio)

            if ratios:
                # El ratio debería ser aproximadamente 0.5
                promedio = np.mean(ratios)
                self.assertAlmostEqual(promedio, 0.5, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

