"""
Método de Bisección para encontrar raíces de ecuaciones no lineales.

El método de bisección es un algoritmo de búsqueda de raíces que divide
repetidamente un intervalo a la mitad y selecciona el subintervalo donde
la función cambia de signo.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Any, Tuple, Optional


class MetodoBiseccion:
    """
    Implementación del método de bisección.

    Attributes:
        f: Función a evaluar.
        a: Extremo izquierdo del intervalo.
        b: Extremo derecho del intervalo.
        tolerancia: Criterio de convergencia.
        max_iter: Número máximo de iteraciones.
        historial: Lista con el historial de iteraciones.
        raiz: Raíz aproximada encontrada.
        convergencia: Indica si el método convergió.
        tiempo_ejecucion: Tiempo de ejecución en segundos.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tolerancia: float = 1e-6,
        max_iter: int = 100
    ):
        """
        Inicializa el método de bisección.

        Args:
            f: Función cuya raíz se desea encontrar.
            a: Extremo izquierdo del intervalo.
            b: Extremo derecho del intervalo.
            tolerancia: Criterio de convergencia (default: 1e-6).
            max_iter: Número máximo de iteraciones (default: 100).
        """
        self.f = f
        self.a = a
        self.b = b
        self.tolerancia = tolerancia
        self.max_iter = max_iter

        self.historial: List[Dict[str, Any]] = []
        self.raiz: Optional[float] = None
        self.convergencia: bool = False
        self.tiempo_ejecucion: float = 0.0
        self.mensaje: str = ""
        self.iteraciones: int = 0
        self.error_final: float = float('inf')
        self.evaluaciones: int = 0

    def validar(self) -> Tuple[bool, str]:
        """
        Valida las condiciones necesarias para ejecutar el método.

        Returns:
            Tupla (es_valido, mensaje).
        """
        # Validar intervalo
        if self.a >= self.b:
            return False, f"Intervalo inválido: [{self.a}, {self.b}]. a debe ser menor que b."

        # Evaluar función en los extremos
        try:
            fa = self.f(self.a)
            fb = self.f(self.b)
            self.evaluaciones += 2
        except Exception as e:
            return False, f"Error al evaluar la función: {str(e)}"

        # Verificar valores numéricos válidos
        if np.isnan(fa) or np.isnan(fb):
            return False, "La función produce valores NaN en los extremos."

        if np.isinf(fa) or np.isinf(fb):
            return False, "La función produce valores infinitos en los extremos."

        # Verificar cambio de signo
        if fa * fb > 0:
            return False, f"No hay cambio de signo en [{self.a}, {self.b}]. " \
                         f"f({self.a})={fa:.6e}, f({self.b})={fb:.6e}"

        return True, "Validación exitosa. Existe cambio de signo en el intervalo."

    def resolver(self) -> Dict[str, Any]:
        """
        Ejecuta el método de bisección.

        Returns:
            Diccionario con los resultados:
            - raiz: Raíz aproximada
            - iteraciones: Número de iteraciones
            - convergencia: Si convergió o no
            - error_final: Error en la última iteración
            - historial: Lista de iteraciones
            - tiempo: Tiempo de ejecución
            - mensaje: Mensaje descriptivo
            - evaluaciones: Número de evaluaciones de la función
        """
        # Validar condiciones
        es_valido, mensaje = self.validar()
        if not es_valido:
            self.mensaje = mensaje
            return self._crear_resultado()

        inicio = time.perf_counter()

        a = self.a
        b = self.b
        fa = self.f(a)
        fb = self.f(b)
        c_anterior = a

        for n in range(1, self.max_iter + 1):
            # Calcular punto medio
            c = (a + b) / 2
            fc = self.f(c)
            self.evaluaciones += 1

            # Calcular errores
            # Error absoluto según la guía: |xₙ − xₙ₋₁|
            error_absoluto = abs(c - c_anterior)
            # Cota del error del método (para referencia): |b-a|/2
            cota_error = abs(b - a) / 2
            error_relativo = abs(c - c_anterior) / abs(c) if c != 0 else abs(c - c_anterior)
            error_relativo_pct = error_relativo * 100  # Porcentaje

            # Guardar iteración en historial
            self.historial.append({
                'n': n,
                'a': a,
                'b': b,
                'c': c,
                'f(a)': fa,
                'f(b)': fb,
                'f(c)': fc,
                'error_absoluto': error_absoluto,
                'cota_error': cota_error,
                'error_relativo': error_relativo,
                'error_rel_%': error_relativo_pct
            })

            # Verificar convergencia
            if abs(fc) < self.tolerancia or error_absoluto < self.tolerancia:
                self.raiz = c
                self.convergencia = True
                self.iteraciones = n
                self.error_final = error_absoluto
                self.mensaje = f"Convergencia alcanzada en {n} iteraciones."
                break

            # Determinar nuevo intervalo
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            c_anterior = c

        else:
            # No convergió en max_iter iteraciones
            self.raiz = c
            self.convergencia = False
            self.iteraciones = self.max_iter
            self.error_final = error_absoluto
            self.mensaje = f"No convergió en {self.max_iter} iteraciones."

        self.tiempo_ejecucion = time.perf_counter() - inicio

        return self._crear_resultado()

    def _crear_resultado(self) -> Dict[str, Any]:
        """
        Crea el diccionario de resultados.

        Returns:
            Diccionario con todos los resultados del método.
        """
        return {
            'raiz': self.raiz,
            'iteraciones': self.iteraciones,
            'convergencia': self.convergencia,
            'error_final': self.error_final,
            'historial': self.historial,
            'tiempo': self.tiempo_ejecucion,
            'mensaje': self.mensaje,
            'evaluaciones': self.evaluaciones,
            'metodo': 'Bisección'
        }

    def obtener_datos_grafica(self) -> Dict[str, Any]:
        """
        Obtiene los datos necesarios para graficar.

        Returns:
            Diccionario con datos para graficación.
        """
        if not self.historial:
            return {}

        iteraciones = [h['n'] for h in self.historial]
        errores = [h['error_absoluto'] for h in self.historial]
        aproximaciones = [h['c'] for h in self.historial]

        return {
            'iteraciones': iteraciones,
            'errores': errores,
            'aproximaciones': aproximaciones,
            'raiz': self.raiz,
            'intervalo': (self.a, self.b)
        }

