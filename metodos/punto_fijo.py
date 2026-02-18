"""
Método de Punto Fijo para encontrar raíces de ecuaciones no lineales.

El método de punto fijo transforma f(x) = 0 en x = g(x) y busca
el punto donde x = g(x).
"""

import time
import numpy as np
from typing import Callable, Dict, List, Any, Tuple, Optional


class MetodoPuntoFijo:
    """
    Implementación del método de punto fijo.

    Transforma el problema f(x) = 0 en encontrar x tal que x = g(x).

    Attributes:
        g: Función de iteración g(x).
        dg: Derivada de g(x) para verificar convergencia.
        x0: Valor inicial.
        tolerancia: Criterio de convergencia.
        max_iter: Número máximo de iteraciones.
    """

    def __init__(
        self,
        g: Callable[[float], float],
        x0: float,
        tolerancia: float = 1e-8,
        max_iter: int = 100,
        dg: Optional[Callable[[float], float]] = None
    ):
        """
        Inicializa el método de punto fijo.

        Args:
            g: Función de iteración g(x) donde x = g(x).
            x0: Valor inicial.
            tolerancia: Criterio de convergencia (default: 1e-8).
            max_iter: Número máximo de iteraciones (default: 100).
            dg: Derivada de g(x) para verificar condición de convergencia.
        """
        self.g = g
        self.x0 = x0
        self.tolerancia = tolerancia
        self.max_iter = max_iter
        self.dg = dg

        self.historial: List[Dict[str, Any]] = []
        self.raiz: Optional[float] = None
        self.convergencia: bool = False
        self.tiempo_ejecucion: float = 0.0
        self.mensaje: str = ""
        self.iteraciones: int = 0
        self.error_final: float = float('inf')
        self.evaluaciones: int = 0
        self.divergencia_detectada: bool = False
        self.condicion_g_prima: Optional[float] = None

    def verificar_condicion_convergencia(self, x: float) -> Tuple[bool, float]:
        """
        Verifica si |g'(x)| < 1 para garantizar convergencia.

        Args:
            x: Punto donde verificar la condición.

        Returns:
            Tupla (cumple_condicion, valor_derivada).
        """
        if self.dg is None:
            return True, 0.0  # No se puede verificar sin derivada

        try:
            dg_x = self.dg(x)
            self.condicion_g_prima = abs(dg_x)
            return abs(dg_x) < 1, dg_x
        except:
            return True, 0.0

    def validar(self) -> Tuple[bool, str]:
        """
        Valida las condiciones iniciales.

        Returns:
            Tupla (es_valido, mensaje).
        """
        try:
            g_x0 = self.g(self.x0)
            self.evaluaciones += 1
        except Exception as e:
            return False, f"Error al evaluar g(x) en x0={self.x0}: {str(e)}"

        if np.isnan(g_x0) or np.isinf(g_x0):
            return False, f"g({self.x0}) produce un valor inválido: {g_x0}"

        # Verificar condición de convergencia si se tiene la derivada
        if self.dg is not None:
            cumple, dg_x0 = self.verificar_condicion_convergencia(self.x0)
            if not cumple:
                return False, f"|g'({self.x0})| = {abs(dg_x0):.6f} >= 1. " \
                             f"El método puede no converger."

        return True, "Validación exitosa."

    def resolver(self) -> Dict[str, Any]:
        """
        Ejecuta el método de punto fijo.

        Returns:
            Diccionario con los resultados del método.
        """
        es_valido, mensaje = self.validar()
        if not es_valido:
            self.mensaje = mensaje
            return self._crear_resultado()

        inicio = time.perf_counter()

        x_n = self.x0

        for n in range(1, self.max_iter + 1):
            try:
                g_xn = self.g(x_n)
                self.evaluaciones += 1
            except Exception as e:
                self.mensaje = f"Error en iteración {n}: {str(e)}"
                self.divergencia_detectada = True
                self.raiz = x_n  # Guardar última aproximación
                self.iteraciones = n - 1
                break

            # Verificar valores válidos
            if np.isnan(g_xn) or np.isinf(g_xn):
                self.mensaje = f"Divergencia detectada en iteración {n}: g({x_n}) = {g_xn}"
                self.divergencia_detectada = True
                self.raiz = x_n  # Guardar última aproximación válida
                self.iteraciones = n - 1
                break

            # Detectar divergencia por valores muy grandes
            if abs(g_xn) > 1e10:
                self.mensaje = f"Divergencia detectada: |g(x)| > 1e10"
                self.divergencia_detectada = True
                self.raiz = x_n  # Guardar última aproximación válida
                self.iteraciones = n - 1
                break

            # Calcular errores
            error_absoluto = abs(g_xn - x_n)
            error_relativo = error_absoluto / abs(g_xn) if g_xn != 0 else error_absoluto
            error_relativo_pct = error_relativo * 100  # Porcentaje

            # Guardar iteración
            self.historial.append({
                'n': n,
                'x_n': x_n,
                'g(x_n)': g_xn,
                'error_absoluto': error_absoluto,
                'error_relativo': error_relativo,
                'error_rel_%': error_relativo_pct
            })

            # Verificar convergencia
            if error_absoluto < self.tolerancia:
                self.raiz = g_xn
                self.convergencia = True
                self.iteraciones = n
                self.error_final = error_absoluto
                self.mensaje = f"Convergencia alcanzada en {n} iteraciones."
                break

            # Siguiente iteración
            x_n = g_xn

        else:
            if not self.divergencia_detectada:
                self.raiz = x_n
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
            'metodo': 'Punto Fijo',
            'divergencia': self.divergencia_detectada,
            'condicion_g_prima': self.condicion_g_prima
        }

    def obtener_datos_grafica(self) -> Dict[str, Any]:
        """Obtiene los datos necesarios para graficar."""
        if not self.historial:
            return {}

        iteraciones = [h['n'] for h in self.historial]
        errores = [h['error_absoluto'] for h in self.historial]
        x_valores = [self.x0] + [h['g(x_n)'] for h in self.historial]

        # Datos para cobweb plot
        cobweb_x = [self.x0]
        cobweb_y = [0]

        for h in self.historial:
            x = h['x_n']
            gx = h['g(x_n)']
            # Línea vertical a g(x)
            cobweb_x.append(x)
            cobweb_y.append(gx)
            # Línea horizontal a y=x
            cobweb_x.append(gx)
            cobweb_y.append(gx)

        return {
            'iteraciones': iteraciones,
            'errores': errores,
            'aproximaciones': x_valores,
            'raiz': self.raiz,
            'cobweb_x': cobweb_x,
            'cobweb_y': cobweb_y,
            'x0': self.x0
        }

