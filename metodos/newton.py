"""
Método de Newton-Raphson para encontrar raíces de ecuaciones no lineales.

El método de Newton-Raphson utiliza la derivada de la función para
obtener una convergencia cuadrática hacia la raíz.
"""

import time
import numpy as np
from typing import Callable, Dict, List, Any, Tuple, Optional


class MetodoNewton:
    """
    Implementación del método de Newton-Raphson.

    Utiliza la fórmula iterativa:
    x_{n+1} = x_n - f(x_n) / f'(x_n)

    Attributes:
        f: Función a evaluar.
        df: Derivada de la función.
        x0: Valor inicial.
        tolerancia: Criterio de convergencia.
        max_iter: Número máximo de iteraciones.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        tolerancia: float = 1e-10,
        max_iter: int = 100
    ):
        """
        Inicializa el método de Newton-Raphson.

        Args:
            f: Función cuya raíz se desea encontrar.
            df: Derivada de la función f.
            x0: Valor inicial.
            tolerancia: Criterio de convergencia (default: 1e-10).
            max_iter: Número máximo de iteraciones (default: 100).
        """
        self.f = f
        self.df = df
        self.x0 = x0
        self.tolerancia = tolerancia
        self.max_iter = max_iter

        self.historial: List[Dict[str, Any]] = []
        self.raiz: Optional[float] = None
        self.convergencia: bool = False
        self.tiempo_ejecucion: float = 0.0
        self.mensaje: str = ""
        self.iteraciones: int = 0
        self.error_final: float = float('inf')
        self.evaluaciones_f: int = 0
        self.evaluaciones_df: int = 0
        self.tangentes: List[Dict[str, Any]] = []  # Para graficar tangentes

    def validar(self) -> Tuple[bool, str]:
        """
        Valida las condiciones iniciales.

        Returns:
            Tupla (es_valido, mensaje).
        """
        try:
            f_x0 = self.f(self.x0)
            df_x0 = self.df(self.x0)
            self.evaluaciones_f += 1
            self.evaluaciones_df += 1
        except Exception as e:
            return False, f"Error al evaluar la función en x0={self.x0}: {str(e)}"

        if np.isnan(f_x0) or np.isnan(df_x0):
            return False, f"La función o su derivada produce NaN en x0={self.x0}."

        if np.isinf(f_x0) or np.isinf(df_x0):
            return False, f"La función o su derivada produce infinito en x0={self.x0}."

        if abs(df_x0) < 1e-12:
            # Si f(x0) también es pequeño, podríamos estar cerca de una raíz múltiple
            if abs(f_x0) < self.tolerancia:
                # Estamos en la raíz
                self.raiz = self.x0
                self.convergencia = True
                return True, f"x0={self.x0} ya es una raíz (múltiple): f(x0)≈0 y f'(x0)≈0"
            # Intentar perturbación
            return False, f"La derivada es casi cero en x0={self.x0}: f'({self.x0})={df_x0:.2e}. Intente con otro valor inicial."

        return True, "Validación exitosa."

    def resolver(self) -> Dict[str, Any]:
        """
        Ejecuta el método de Newton-Raphson.

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
                f_xn = self.f(x_n)
                df_xn = self.df(x_n)
                self.evaluaciones_f += 1
                self.evaluaciones_df += 1
            except Exception as e:
                self.mensaje = f"Error en iteración {n}: {str(e)}"
                self.raiz = x_n  # Guardar última aproximación
                self.iteraciones = n - 1
                break

            # Verificar división por cero
            if abs(df_xn) < 1e-15:
                # Si f(x) también es muy pequeño, podríamos estar en una raíz múltiple
                if abs(f_xn) < self.tolerancia:
                    self.raiz = x_n
                    self.convergencia = True
                    self.iteraciones = n
                    self.error_final = abs(f_xn)
                    self.mensaje = f"Raíz múltiple encontrada en {n} iteraciones: f({x_n})≈0 y f'({x_n})≈0"
                    break
                # Intentar una pequeña perturbación para escapar del punto crítico
                perturbacion = 1e-8
                x_n_perturbado = x_n + perturbacion
                df_perturbado = self.df(x_n_perturbado)
                self.evaluaciones_df += 1
                if abs(df_perturbado) > 1e-15:
                    # Usar el valor perturbado
                    x_n = x_n_perturbado
                    df_xn = df_perturbado
                    f_xn = self.f(x_n)
                    self.evaluaciones_f += 1
                else:
                    self.mensaje = f"Derivada casi cero en iteración {n}: f'({x_n})={df_xn:.2e}"
                    self.raiz = x_n  # Guardar última aproximación
                    self.iteraciones = n - 1
                    break

            # Calcular siguiente aproximación
            x_n1 = x_n - f_xn / df_xn

            # Guardar datos de tangente para gráfica
            self.tangentes.append({
                'x': x_n,
                'f(x)': f_xn,
                "f'(x)": df_xn,
                'x_siguiente': x_n1
            })

            # Calcular errores
            error_absoluto = abs(x_n1 - x_n)
            error_relativo = error_absoluto / abs(x_n1) if x_n1 != 0 else error_absoluto
            error_relativo_pct = error_relativo * 100  # Porcentaje

            # Guardar iteración
            self.historial.append({
                'n': n,
                'x_n': x_n,
                'f(x_n)': f_xn,
                "f'(x_n)": df_xn,
                'x_n+1': x_n1,
                'error_absoluto': error_absoluto,
                'error_relativo': error_relativo,
                'error_rel_%': error_relativo_pct
            })

            # Verificar convergencia
            if error_absoluto < self.tolerancia or abs(f_xn) < self.tolerancia:
                self.raiz = x_n1
                self.convergencia = True
                self.iteraciones = n
                self.error_final = error_absoluto
                self.mensaje = f"Convergencia alcanzada en {n} iteraciones."
                break

            # Detectar divergencia
            if abs(x_n1) > 1e15:
                self.mensaje = f"Divergencia detectada en iteración {n}."
                self.raiz = x_n  # Guardar última aproximación válida
                self.iteraciones = n
                self.error_final = error_absoluto
                break

            x_n = x_n1

        else:
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
            'evaluaciones': self.evaluaciones_f + self.evaluaciones_df,
            'evaluaciones_f': self.evaluaciones_f,
            'evaluaciones_df': self.evaluaciones_df,
            'metodo': 'Newton-Raphson',
            'tangentes': self.tangentes
        }

    def obtener_datos_grafica(self) -> Dict[str, Any]:
        """Obtiene los datos necesarios para graficar."""
        if not self.historial:
            return {}

        iteraciones = [h['n'] for h in self.historial]
        errores = [h['error_absoluto'] for h in self.historial]
        aproximaciones = [self.x0] + [h['x_n+1'] for h in self.historial]

        return {
            'iteraciones': iteraciones,
            'errores': errores,
            'aproximaciones': aproximaciones,
            'raiz': self.raiz,
            'tangentes': self.tangentes,
            'x0': self.x0
        }

    def verificar_convergencia_cuadratica(self) -> Dict[str, Any]:
        """
        Verifica si la convergencia fue cuadrática.

        La convergencia cuadrática implica que:
        e_{n+1} ≈ C * e_n^2

        Returns:
            Diccionario con el análisis de convergencia.
        """
        if len(self.historial) < 3:
            return {'verificable': False, 'mensaje': 'Se necesitan al menos 3 iteraciones.'}

        errores = [h['error_absoluto'] for h in self.historial]
        ratios = []

        for i in range(1, len(errores)):
            if errores[i-1] > 0:
                # Para convergencia cuadrática: e_{n+1}/e_n^2 ≈ constante
                ratio = errores[i] / (errores[i-1] ** 2) if errores[i-1] > 1e-15 else 0
                ratios.append(ratio)

        return {
            'verificable': True,
            'ratios': ratios,
            'promedio_ratio': np.mean(ratios) if ratios else 0,
            'mensaje': 'La convergencia cuadrática se caracteriza por ratios constantes.'
        }

