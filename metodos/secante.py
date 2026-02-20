"""
Método de la Secante para encontrar raíces de ecuaciones no lineales.

El método de la secante es similar a Newton-Raphson pero no requiere
la derivada, usando una aproximación con dos puntos.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Any, Tuple, Optional


class MetodoSecante:
    """
    Implementación del método de la secante.

    Utiliza la fórmula iterativa:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Attributes:
        f: Función a evaluar.
        x0: Primer valor inicial.
        x1: Segundo valor inicial.
        tolerancia: Criterio de convergencia.
        max_iter: Número máximo de iteraciones.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tolerancia: float = 1e-9,
        max_iter: int = 100
    ):
        """
        Inicializa el método de la secante.

        Args:
            f: Función cuya raíz se desea encontrar.
            x0: Primer valor inicial.
            x1: Segundo valor inicial.
            tolerancia: Criterio de convergencia (default: 1e-9).
            max_iter: Número máximo de iteraciones (default: 100).
        """
        self.f = f
        self.x0 = x0
        self.x1 = x1
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
        self.secantes: List[Dict[str, Any]] = []  # Para graficar secantes

    def validar(self) -> Tuple[bool, str]:
        """
        Valida las condiciones iniciales.

        Returns:
            Tupla (es_valido, mensaje).
        """
        if self.x0 == self.x1:
            return False, "Los valores iniciales x0 y x1 deben ser diferentes."

        try:
            f_x0 = self.f(self.x0)
            f_x1 = self.f(self.x1)
            self.evaluaciones += 2
        except Exception as e:
            return False, f"Error al evaluar la función: {str(e)}"

        if np.isnan(f_x0) or np.isnan(f_x1):
            return False, "La función produce valores NaN en los puntos iniciales."

        if np.isinf(f_x0) or np.isinf(f_x1):
            return False, "La función produce valores infinitos en los puntos iniciales."

        if abs(f_x1 - f_x0) < 1e-15:
            return False, f"f(x1) ≈ f(x0), posible división por cero."

        return True, "Validación exitosa."

    def resolver(self) -> Dict[str, Any]:
        """
        Ejecuta el método de la secante.

        Returns:
            Diccionario con los resultados del método.
        """
        es_valido, mensaje = self.validar()
        if not es_valido:
            self.mensaje = mensaje
            return self._crear_resultado()

        inicio = time.perf_counter()

        x_prev = self.x0
        x_curr = self.x1
        f_prev = self.f(x_prev)
        f_curr = self.f(x_curr)

        for n in range(1, self.max_iter + 1):
            # Verificar división por cero
            denominador = f_curr - f_prev
            if abs(denominador) < 1e-15:
                self.mensaje = f"División por cero en iteración {n}: f(x_n) ≈ f(x_{n-1})"
                self.raiz = x_curr  # Guardar última aproximación
                self.iteraciones = n - 1
                break

            # Calcular siguiente aproximación
            x_next = x_curr - f_curr * (x_curr - x_prev) / denominador

            try:
                f_next = self.f(x_next)
                self.evaluaciones += 1
            except Exception as e:
                self.mensaje = f"Error al evaluar f({x_next}): {str(e)}"
                self.raiz = x_curr  # Guardar última aproximación
                self.iteraciones = n - 1
                break

            # Guardar datos de secante para gráfica
            self.secantes.append({
                'x_prev': x_prev,
                'x_curr': x_curr,
                'f_prev': f_prev,
                'f_curr': f_curr,
                'x_next': x_next
            })

            # Calcular errores
            error_absoluto = abs(x_next - x_curr)
            error_relativo = error_absoluto / abs(x_next) if x_next != 0 else error_absoluto
            error_relativo_pct = error_relativo * 100  # Porcentaje

            # Guardar iteración
            self.historial.append({
                'n': n,
                'x_{n-1}': x_prev,
                'x_n': x_curr,
                'f(x_{n-1})': f_prev,
                'f(x_n)': f_curr,
                'x_{n+1}': x_next,
                'error_absoluto': error_absoluto,
                'error_relativo': error_relativo,
                'error_rel_%': error_relativo_pct
            })

            # Verificar convergencia
            if error_absoluto < self.tolerancia or abs(f_next) < self.tolerancia:
                self.raiz = x_next
                self.convergencia = True
                self.iteraciones = n
                self.error_final = error_absoluto
                self.mensaje = f"Convergencia alcanzada en {n} iteraciones."
                break

            # Verificar divergencia
            if abs(x_next) > 1e15 or np.isnan(f_next) or np.isinf(f_next):
                self.mensaje = f"Divergencia detectada en iteración {n}."
                self.raiz = x_curr  # Guardar última aproximación válida
                self.iteraciones = n
                self.error_final = error_absoluto
                break

            # Actualizar para siguiente iteración
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next

        else:
            self.raiz = x_curr
            self.convergencia = False
            self.iteraciones = self.max_iter
            self.error_final = error_absoluto if 'error_absoluto' in locals() else float('inf')
            self.mensaje = f"No convergió en {self.max_iter} iteraciones."

        self.tiempo_ejecucion = time.perf_counter() - inicio

        return self._crear_resultado()

    def _crear_resultado(self) -> Dict[str, Any]:
        """Crea el diccionario de resultados."""
        return {
            'raiz': self.raiz,
            'iteraciones': self.iteraciones,
            'convergencia': self.convergencia,
            'error_final': self.error_final,
            'historial': self.historial,
            'tiempo': self.tiempo_ejecucion,
            'mensaje': self.mensaje,
            'evaluaciones': self.evaluaciones,
            'metodo': 'Secante',
            'secantes': self.secantes
        }

    def obtener_datos_grafica(self) -> Dict[str, Any]:
        """Obtiene los datos necesarios para graficar."""
        if not self.historial:
            return {}

        iteraciones = [h['n'] for h in self.historial]
        errores = [h['error_absoluto'] for h in self.historial]
        aproximaciones = [self.x0, self.x1] + [h['x_{n+1}'] for h in self.historial]

        return {
            'iteraciones': iteraciones,
            'errores': errores,
            'aproximaciones': aproximaciones,
            'raiz': self.raiz,
            'secantes': self.secantes,
            'x0': self.x0,
            'x1': self.x1
        }

