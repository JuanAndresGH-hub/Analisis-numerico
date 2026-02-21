"""
Definiciones de funciones matemáticas para los ejercicios.

Este módulo contiene las funciones específicas de cada ejercicio,
así como utilidades para obtener funciones y sus derivadas.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional
import sympy as sp


# Símbolo para cálculos simbólicos
x = sp.Symbol('x')


# Definiciones de funciones para cada ejercicio
FUNCIONES_EJERCICIOS: Dict[str, Dict[str, Any]] = {
    "Ejercicio 1 - Bisección": {
        "nombre": "T'(λ) = 1.6λ - 3.2 + 1/(λ+1) [Optimización Hash Table]",
        "funcion_str": "1.6*x - 3.2 + 1/(x + 1)",
        "funcion_original": "2.5 + 0.8*x**2 - 3.2*x + log(x + 1)",
        "intervalo": (0.5, 2.5),
        "tolerancia": 1e-6,
        "max_iter": 100,
        "metodo": "biseccion",
        "descripcion": "Encontrar λ óptimo que minimiza T(λ) = 2.5 + 0.8λ² - 3.2λ + ln(λ+1). Se busca la raíz de T'(λ).",
        "interpretacion": "El valor λ encontrado representa el factor de carga óptimo para la hash table. Un factor de carga menor a 1 indica que la tabla debe tener más slots que elementos, reduciendo colisiones. Valores típicos óptimos están entre 0.5 y 0.75 para hash tables eficientes."
    },
    "Ejercicio 2 - Falsa Posición": {
        "nombre": "E(x) = x³ - 6x² + 11x - 6.5 [Balanceo de Carga]",
        "funcion_str": "x**3 - 6*x**2 + 11*x - 6.5",
        "intervalo": (2, 4),
        "tolerancia": 1e-7,
        "max_iter": 100,
        "metodo": "falsa_posicion",
        "descripcion": "Encontrar el número óptimo de workers activos para balanceo de carga en sistema distribuido.",
        "interpretacion": "El valor x encontrado representa el número óptimo de workers activos en el sistema distribuido. Este valor maximiza la eficiencia del balanceo de carga. En la práctica, se redondea al entero más cercano para determinar cuántos workers deben estar activos simultáneamente.",
        "comparar_con_biseccion": True
    },
    "Ejercicio 3 - Punto Fijo": {
        "nombre": "x = 0.5cos(x) + 1.5 [Crecimiento Base de Datos]",
        "funcion_str": "0.5*cos(x) + 1.5",  # g(x)
        "derivada_g_str": "-0.5*sin(x)",  # g'(x) para verificar convergencia
        "funcion_original": "x - 0.5*cos(x) - 1.5",  # f(x) = x - g(x)
        "x0": 1.0,
        "x0_alternativos": [0.5, 1.0, 1.5, 2.0],
        "tolerancia": 1e-8,
        "max_iter": 100,
        "metodo": "punto_fijo",
        "descripcion": "Predecir cuándo la base de datos SaaS alcanzará el 80% de capacidad. x representa meses desde el inicio.",
        "interpretacion": "El valor x encontrado representa el número de meses desde el inicio del sistema cuando la base de datos alcanzará el 80% de su capacidad. Este valor permite planificar con anticipación la expansión de almacenamiento o migración a una infraestructura más grande.",
        "condicion_convergencia": "|g'(x)| = |−0.5·sin(x)| ≤ 0.5 < 1, por lo que el método SIEMPRE converge para cualquier x₀."
    },
    "Ejercicio 4 - Newton-Raphson": {
        "nombre": "T(n) = n³ - 8n² + 20n - 16 [Análisis de Concurrencia]",
        "funcion_str": "x**3 - 8*x**2 + 20*x - 16",
        "derivada_str": "3*x**2 - 16*x + 20",
        "derivada_latex": "T'(n) = 3n² - 16n + 20",
        "x0": 3.0,  # Valor por defecto según la guía
        "x0_alternativos": [1.0, 2.0, 3.0, 5.0],  # Valores de la guía (incluye n=2 problemático)
        "tolerancia": 1e-10,
        "max_iter": 100,
        "metodo": "newton",
        "descripcion": "Determinar el número óptimo de threads donde el overhead de sincronización equilibra el beneficio del paralelismo.",
        "interpretacion": "El valor n encontrado representa el número óptimo de threads para el sistema de procesamiento paralelo. ADVERTENCIA: Esta función tiene una raíz DOBLE en n=2 donde f'(2)=0, lo que hace que Newton-Raphson converja lentamente o falle. La raíz simple está en n=4. Para n₀=3.0, el método salta a n=2 donde la derivada es cero. Use n₀=5.0 para converger correctamente a n=4.",
        "convergencia_cuadratica": "Newton-Raphson tiene convergencia cuadrática para raíces simples. Para la raíz doble en n=2, la convergencia es LINEAL. En escala logarítmica, log(eₙ₊₁) ≈ 2·log(eₙ) + C para raíces simples.",
        "nota_raiz_multiple": "T(n) = (n-2)²(n-4). Raíz doble en n=2, raíz simple en n=4.",
        "advertencia_n2": "Si n₀=2.0 o n₀=3.0, el método puede fallar porque f'(2)=0. Se recomienda usar n₀=5.0 para convergencia estable."
    },
    "Ejercicio 5 - Secante": {
        "nombre": "P(x) = x·e^(-x/2) - 0.3 [Predicción de Escalabilidad Cloud]",
        "funcion_str": "x*exp(-x/2) - 0.3",
        "derivada_str": "exp(-x/2)*(1 - x/2)",  # d/dx[x·e^(-x/2)] = e^(-x/2)·(1 - x/2)
        "derivada_latex": "P'(x) = e^{-x/2}·(1 - x/2)",
        "x0": 0.5,
        "x1": 1.0,
        "tolerancia": 1e-9,
        "max_iter": 100,
        "metodo": "secante",
        "descripcion": "Encontrar el punto donde el costo de infraestructura cloud iguala los ingresos. x representa miles de usuarios activos.",
        "interpretacion": "El valor x encontrado representa el número de miles de usuarios activos donde el modelo financiero P(x) = x·e^(-x/2) - 0.3 = 0. Este es el punto de equilibrio: con menos usuarios hay pérdidas, con más usuarios hay ganancias. El método de la secante es ideal aquí porque evita calcular la derivada analíticamente.",
        "comparar_con_newton": True,
        "analisis_costo": "La derivada P'(x) = e^(-x/2)·(1 - x/2) no es trivial de calcular a mano, pero es manejable. Sin embargo, si la función fuera más compleja (ej: datos empíricos), la secante sería preferible.",
        "orden_convergencia": "Secante: orden φ ≈ 1.618 (número áureo). Newton: orden 2 (cuadrática). Secante requiere ~1.44 veces más iteraciones pero sin derivadas."
    }
}


def obtener_funcion(funcion_str: str) -> Callable[[float], float]:
    """
    Convierte una cadena de función matemática a una función evaluable.

    Args:
        funcion_str: Cadena representando la función (ej: "x**2 - 2").

    Returns:
        Función callable que evalúa la expresión.

    Raises:
        ValueError: Si la cadena no puede ser convertida a función.
    """
    try:
        # Convertir a expresión sympy
        expr = sp.sympify(funcion_str)

        # Convertir a función numérica usando lambdify
        func = sp.lambdify(x, expr, modules=['numpy'])

        return func
    except Exception as e:
        raise ValueError(f"No se pudo convertir '{funcion_str}' a función: {str(e)}")


def obtener_derivada(funcion_str: str) -> Callable[[float], float]:
    """
    Calcula la derivada simbólica de una función y la convierte a callable.

    Args:
        funcion_str: Cadena representando la función.

    Returns:
        Función callable que evalúa la derivada.

    Raises:
        ValueError: Si no se puede calcular la derivada.
    """
    try:
        # Convertir a expresión sympy
        expr = sp.sympify(funcion_str)

        # Calcular derivada
        derivada = sp.diff(expr, x)

        # Convertir a función numérica
        func_derivada = sp.lambdify(x, derivada, modules=['numpy'])

        return func_derivada
    except Exception as e:
        raise ValueError(f"No se pudo calcular la derivada de '{funcion_str}': {str(e)}")


def obtener_derivada_str(funcion_str: str) -> str:
    """
    Obtiene la representación en cadena de la derivada.

    Args:
        funcion_str: Cadena representando la función.

    Returns:
        Cadena representando la derivada.
    """
    try:
        expr = sp.sympify(funcion_str)
        derivada = sp.diff(expr, x)
        return str(derivada)
    except Exception as e:
        raise ValueError(f"No se pudo calcular la derivada: {str(e)}")


def evaluar_funcion_seguro(f: Callable[[float], float], valor: float) -> Optional[float]:
    """
    Evalúa una función de manera segura, manejando excepciones.

    Args:
        f: Función a evaluar.
        valor: Valor en el cual evaluar la función.

    Returns:
        Resultado de la evaluación o None si hay error.
    """
    try:
        resultado = f(valor)
        if np.isnan(resultado) or np.isinf(resultado):
            return None
        return resultado
    except:
        return None


def obtener_info_ejercicio(nombre_ejercicio: str) -> Dict[str, Any]:
    """
    Obtiene la información completa de un ejercicio.

    Args:
        nombre_ejercicio: Nombre del ejercicio.

    Returns:
        Diccionario con la información del ejercicio.

    Raises:
        KeyError: Si el ejercicio no existe.
    """
    if nombre_ejercicio not in FUNCIONES_EJERCICIOS:
        raise KeyError(f"Ejercicio '{nombre_ejercicio}' no encontrado.")

    return FUNCIONES_EJERCICIOS[nombre_ejercicio].copy()


def listar_ejercicios() -> list:
    """
    Lista los nombres de todos los ejercicios disponibles.

    Returns:
        Lista de nombres de ejercicios.
    """
    return list(FUNCIONES_EJERCICIOS.keys())

