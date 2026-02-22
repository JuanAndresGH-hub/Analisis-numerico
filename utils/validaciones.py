"""
Módulo de validaciones para los métodos numéricos.

Este módulo contiene funciones de validación para verificar
las condiciones necesarias antes de ejecutar los métodos numéricos.

Funciones disponibles:
    - validar_intervalo: Valida que a < b
    - validar_cambio_signo: Verifica cambio de signo en [a,b]
    - validar_tolerancia: Verifica tolerancia válida
    - validar_max_iteraciones: Verifica número de iteraciones
    - validar_derivada_no_cero: Verifica f'(x) ≠ 0
    - validar_condicion_punto_fijo: Verifica |g'(x)| < 1
    - validar_division_por_cero: Previene división por cero
    - validar_valor_inicial: Verifica valores iniciales
    - validar_funcion_evaluable: Verifica evaluación segura
    - validar_convergencia: Detecta divergencia
    - calcular_orden_convergencia: Estima orden de convergencia

Autor: Proyecto de Análisis Numérico
Fecha: 2024
"""

from typing import Callable, Tuple, Optional, List
import numpy as np


def validar_intervalo(a: float, b: float) -> Tuple[bool, str]:
    """
    Valida que el intervalo sea correcto.

    Args:
        a: Extremo izquierdo del intervalo.
        b: Extremo derecho del intervalo.

    Returns:
        Tupla (es_valido, mensaje).
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False, "Los valores del intervalo deben ser numéricos."

    if np.isnan(a) or np.isnan(b):
        return False, "Los valores del intervalo no pueden ser NaN."

    if np.isinf(a) or np.isinf(b):
        return False, "Los valores del intervalo no pueden ser infinitos."

    if a >= b:
        return False, f"El extremo izquierdo ({a}) debe ser menor que el derecho ({b})."

    return True, "Intervalo válido."


def validar_cambio_signo(f: Callable[[float], float], a: float, b: float) -> Tuple[bool, str]:
    """
    Valida que exista cambio de signo en el intervalo [a, b].

    Args:
        f: Función a evaluar.
        a: Extremo izquierdo del intervalo.
        b: Extremo derecho del intervalo.

    Returns:
        Tupla (existe_cambio, mensaje).
    """
    try:
        fa = f(a)
        fb = f(b)
    except Exception as e:
        return False, f"Error al evaluar la función en los extremos: {str(e)}"

    if np.isnan(fa) or np.isnan(fb):
        return False, "La función produce valores NaN en los extremos del intervalo."

    if np.isinf(fa) or np.isinf(fb):
        return False, "La función produce valores infinitos en los extremos del intervalo."

    if fa * fb > 0:
        return False, f"No hay cambio de signo en [{a}, {b}]. f({a})={fa:.6f}, f({b})={fb:.6f}"

    if fa * fb == 0:
        if fa == 0:
            return True, f"f({a}) = 0, la raíz está en el extremo izquierdo."
        else:
            return True, f"f({b}) = 0, la raíz está en el extremo derecho."

    return True, f"Existe cambio de signo en [{a}, {b}]. f({a})={fa:.6f}, f({b})={fb:.6f}"


def validar_tolerancia(tolerancia: float) -> Tuple[bool, str]:
    """
    Valida que la tolerancia sea un valor positivo razonable.

    Args:
        tolerancia: Valor de tolerancia a validar.

    Returns:
        Tupla (es_valida, mensaje).
    """
    if not isinstance(tolerancia, (int, float)):
        return False, "La tolerancia debe ser un valor numérico."

    if tolerancia <= 0:
        return False, "La tolerancia debe ser un valor positivo."

    if tolerancia >= 1:
        return False, "La tolerancia debe ser menor que 1."

    if tolerancia < 1e-15:
        return False, "La tolerancia es demasiado pequeña (< 1e-15)."

    return True, f"Tolerancia válida: {tolerancia}"


def validar_max_iteraciones(max_iter: int) -> Tuple[bool, str]:
    """
    Valida que el número máximo de iteraciones sea razonable.

    Args:
        max_iter: Número máximo de iteraciones.

    Returns:
        Tupla (es_valido, mensaje).
    """
    if not isinstance(max_iter, int):
        return False, "El número máximo de iteraciones debe ser un entero."

    if max_iter <= 0:
        return False, "El número máximo de iteraciones debe ser positivo."

    if max_iter > 10000:
        return False, "El número máximo de iteraciones no debe exceder 10000."

    return True, f"Máximo de iteraciones válido: {max_iter}"


def validar_derivada_no_cero(
    df: Callable[[float], float],
    x: float,
    umbral: float = 1e-12
) -> Tuple[bool, str]:
    """
    Valida que la derivada no sea cero en el punto dado.

    Args:
        df: Función derivada.
        x: Punto donde evaluar la derivada.
        umbral: Umbral mínimo para considerar la derivada como no cero.

    Returns:
        Tupla (es_valida, mensaje).
    """
    try:
        dfx = df(x)
    except Exception as e:
        return False, f"Error al evaluar la derivada en x={x}: {str(e)}"

    if np.isnan(dfx):
        return False, f"La derivada produce NaN en x={x}."

    if abs(dfx) < umbral:
        return False, f"La derivada es muy cercana a cero en x={x}: f'({x})={dfx:.2e}"

    return True, f"Derivada válida en x={x}: f'({x})={dfx:.6f}"


def validar_condicion_punto_fijo(
    dg: Callable[[float], float],
    x0: float,
    intervalo: Optional[Tuple[float, float]] = None
) -> Tuple[bool, str, Optional[float]]:
    """
    Valida la condición de convergencia del método de punto fijo.

    Verifica que |g'(x)| < 1 en la región de interés.

    Args:
        dg: Derivada de la función g(x).
        x0: Punto inicial.
        intervalo: Intervalo opcional para verificar la condición.

    Returns:
        Tupla (es_valida, mensaje, valor_derivada).
    """
    try:
        dgx0 = dg(x0)
    except Exception as e:
        return False, f"Error al evaluar g'(x) en x={x0}: {str(e)}", None

    if np.isnan(dgx0):
        return False, f"g'(x) produce NaN en x={x0}.", None

    if abs(dgx0) >= 1:
        return False, f"|g'({x0})| = {abs(dgx0):.6f} >= 1. El método puede no converger.", dgx0

    # Verificar en el intervalo si se proporciona
    if intervalo is not None:
        a, b = intervalo
        puntos_prueba = np.linspace(a, b, 20)
        for punto in puntos_prueba:
            try:
                dg_punto = dg(punto)
                if abs(dg_punto) >= 1:
                    return False, f"|g'({punto:.4f})| = {abs(dg_punto):.6f} >= 1. " \
                                  f"El método puede no converger en parte del intervalo.", dg_punto
            except:
                pass

    return True, f"|g'({x0})| = {abs(dgx0):.6f} < 1. Condición de convergencia satisfecha.", dgx0


def validar_division_por_cero(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    umbral: float = 1e-15
) -> Tuple[bool, str]:
    """
    Valida que no haya división por cero en el método de la secante.

    Args:
        f: Función a evaluar.
        x1: Primer punto.
        x2: Segundo punto.
        umbral: Umbral mínimo para el denominador.

    Returns:
        Tupla (es_valida, mensaje).
    """
    try:
        fx1 = f(x1)
        fx2 = f(x2)
    except Exception as e:
        return False, f"Error al evaluar la función: {str(e)}"

    denominador = fx2 - fx1

    if abs(denominador) < umbral:
        return False, f"División por cero: f({x2}) - f({x1}) = {denominador:.2e}"

    return True, "No hay riesgo de división por cero."


def validar_valor_inicial(
    x0: float,
    nombre: str = "x0"
) -> Tuple[bool, str]:
    """
    Valida que un valor inicial sea numérico y finito.

    Args:
        x0: Valor inicial a validar.
        nombre: Nombre del parámetro para el mensaje.

    Returns:
        Tupla (es_valido, mensaje).
    """
    if not isinstance(x0, (int, float)):
        return False, f"El valor {nombre} debe ser numérico."

    if np.isnan(x0):
        return False, f"El valor {nombre} no puede ser NaN."

    if np.isinf(x0):
        return False, f"El valor {nombre} no puede ser infinito."

    return True, f"Valor {nombre} = {x0} válido."


def validar_funcion_evaluable(
    f: Callable[[float], float],
    x: float
) -> Tuple[bool, str, Optional[float]]:
    """
    Valida que una función sea evaluable en un punto dado.

    Args:
        f: Función a evaluar.
        x: Punto donde evaluar.

    Returns:
        Tupla (es_evaluable, mensaje, valor).
    """
    try:
        resultado = f(x)
    except ZeroDivisionError:
        return False, f"División por cero al evaluar f({x}).", None
    except ValueError as e:
        return False, f"Error de valor al evaluar f({x}): {str(e)}", None
    except Exception as e:
        return False, f"Error al evaluar f({x}): {str(e)}", None

    if np.isnan(resultado):
        return False, f"La función produce NaN en x={x}.", None

    if np.isinf(resultado):
        return False, f"La función produce infinito en x={x}.", None

    return True, f"f({x}) = {resultado:.8e}", resultado


def validar_convergencia(
    historial: list,
    umbral_divergencia: float = 1e10
) -> Tuple[bool, str]:
    """
    Analiza el historial de iteraciones para detectar divergencia.

    Args:
        historial: Lista de diccionarios con datos de cada iteración.
        umbral_divergencia: Valor máximo permitido para detectar divergencia.

    Returns:
        Tupla (no_diverge, mensaje).
    """
    if not historial:
        return True, "No hay historial para analizar."

    # Verificar valores crecientes que indican divergencia
    errores = []
    for h in historial:
        if 'error_absoluto' in h:
            errores.append(h['error_absoluto'])

    if len(errores) >= 3:
        # Verificar si los últimos 3 errores están creciendo
        if errores[-1] > errores[-2] > errores[-3]:
            return False, "Se detectó posible divergencia: el error está aumentando."

    # Verificar valores muy grandes
    for h in historial:
        for key, value in h.items():
            if isinstance(value, (int, float)) and abs(value) > umbral_divergencia:
                return False, f"Divergencia detectada: {key} = {value:.2e} excede el umbral."

    return True, "No se detectó divergencia."


def calcular_orden_convergencia(
    errores: list
) -> Tuple[Optional[float], str]:
    """
    Estima el orden de convergencia basado en los errores.

    El orden de convergencia p se estima usando:
    p ≈ log(e_{n+1}/e_n) / log(e_n/e_{n-1})

    Args:
        errores: Lista de errores absolutos.

    Returns:
        Tupla (orden_estimado, descripción).
    """
    if len(errores) < 3:
        return None, "Se necesitan al menos 3 errores para estimar el orden."

    # Filtrar errores válidos (positivos y no muy pequeños)
    errores_validos = [e for e in errores if 0 < e < 1e10 and e > 1e-16]

    if len(errores_validos) < 3:
        return None, "No hay suficientes errores válidos."

    ordenes = []
    for i in range(2, len(errores_validos)):
        e_n = errores_validos[i]
        e_n1 = errores_validos[i-1]
        e_n2 = errores_validos[i-2]

        if e_n1 > 0 and e_n2 > 0:
            ratio1 = e_n / e_n1 if e_n1 > 1e-16 else 0
            ratio2 = e_n1 / e_n2 if e_n2 > 1e-16 else 0

            if ratio1 > 0 and ratio2 > 0 and ratio2 != 1:
                try:
                    p = np.log(ratio1) / np.log(ratio2)
                    if 0.5 < p < 3:  # Filtrar valores razonables
                        ordenes.append(p)
                except:
                    pass

    if not ordenes:
        return None, "No se pudo estimar el orden de convergencia."

    orden_promedio = np.mean(ordenes)

    # Clasificar el orden
    if orden_promedio < 1.2:
        descripcion = f"Convergencia lineal (orden ≈ {orden_promedio:.2f})"
    elif orden_promedio < 1.8:
        descripcion = f"Convergencia superlineal (orden ≈ {orden_promedio:.2f})"
    elif orden_promedio < 2.5:
        descripcion = f"Convergencia cuadrática (orden ≈ {orden_promedio:.2f})"
    else:
        descripcion = f"Convergencia de orden superior (orden ≈ {orden_promedio:.2f})"

    return orden_promedio, descripcion


