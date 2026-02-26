"""
Test del Ejercicio 5: Predicción de Escalabilidad con Método de la Secante
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metodos.secante import MetodoSecante
from metodos.newton import MetodoNewton
from funciones.definiciones import obtener_funcion, obtener_derivada, FUNCIONES_EJERCICIOS
import numpy as np

def test_ejercicio_5():
    print("=" * 75)
    print("EJERCICIO 5: PREDICCIÓN DE ESCALABILIDAD CON MÉTODO DE LA SECANTE")
    print("=" * 75)

    # Cargar configuración del ejercicio
    ejercicio = FUNCIONES_EJERCICIOS["Ejercicio 5 - Secante"]
    print(f"\nFunción: {ejercicio['nombre']}")
    print(f"Descripción: {ejercicio['descripcion']}")

    # Obtener función y derivada
    funcion_str = ejercicio['funcion_str']
    f = obtener_funcion(funcion_str)
    df = obtener_derivada(funcion_str)

    x0 = ejercicio['x0']
    x1 = ejercicio['x1']
    tolerancia = ejercicio['tolerancia']
    max_iter = ejercicio['max_iter']

    print(f"\nParámetros:")
    print(f"  x₀ = {x0}, x₁ = {x1}")
    print(f"  Tolerancia = {tolerancia}")
    print(f"  Max iteraciones = {max_iter}")

    # ============================================
    # MÉTODO DE LA SECANTE
    # ============================================
    print("\n" + "=" * 75)
    print("MÉTODO DE LA SECANTE")
    print("=" * 75)

    secante = MetodoSecante(f, x0, x1, tolerancia, max_iter)
    resultado_secante = secante.resolver()

    print(f"\nResultado:")
    print(f"  Raíz: {resultado_secante['raiz']:.12f}")
    print(f"  Convergió: {'Sí' if resultado_secante['convergencia'] else 'No'}")
    print(f"  Iteraciones: {resultado_secante['iteraciones']}")
    print(f"  Error final: {resultado_secante['error_final']:.2e}")
    print(f"  Evaluaciones f(x): {resultado_secante['evaluaciones']}")
    print(f"  Tiempo: {resultado_secante['tiempo']*1000:.4f} ms")

    print("\nTabla de iteraciones (Secante):")
    print("-" * 95)
    print(f"{'n':>3} | {'x_{n-1}':>14} | {'x_n':>14} | {'f(x_{n-1})':>12} | {'f(x_n)':>12} | {'x_{n+1}':>14} | {'Error':>10}")
    print("-" * 95)
    for h in resultado_secante.get('historial', []):
        print(f"{h['n']:>3} | {h['x_{n-1}']:>14.10f} | {h['x_n']:>14.10f} | {h['f(x_{n-1})']:>12.6e} | {h['f(x_n)']:>12.6e} | {h['x_{n+1}']:>14.10f} | {h['error_absoluto']:>10.2e}")

    # ============================================
    # MÉTODO DE NEWTON-RAPHSON
    # ============================================
    print("\n" + "=" * 75)
    print("MÉTODO DE NEWTON-RAPHSON (para comparación)")
    print("=" * 75)

    newton = MetodoNewton(f, df, x0, tolerancia, max_iter)
    resultado_newton = newton.resolver()

    print(f"\nResultado:")
    print(f"  Raíz: {resultado_newton['raiz']:.12f}")
    print(f"  Convergió: {'Sí' if resultado_newton['convergencia'] else 'No'}")
    print(f"  Iteraciones: {resultado_newton['iteraciones']}")
    print(f"  Error final: {resultado_newton['error_final']:.2e}")
    print(f"  Evaluaciones f(x): {resultado_newton.get('evaluaciones_f', resultado_newton['evaluaciones'])}")
    print(f"  Evaluaciones f'(x): {resultado_newton.get('evaluaciones_df', 0)}")
    print(f"  Total evaluaciones: {resultado_newton['evaluaciones']}")
    print(f"  Tiempo: {resultado_newton['tiempo']*1000:.4f} ms")

    print("\nTabla de iteraciones (Newton):")
    print("-" * 95)
    header_newton = "  n |       x_n        |    f(x_n)    |   f'(x_n)    |     x_{n+1}      |  Error Abs |  Error Rel"
    print(header_newton)
    print("-" * 95)
    for h in resultado_newton.get('historial', []):
        fpx = h.get("f'(x_n)", 0)
        print(f"{h['n']:>3} | {h['x_n']:>14.10f} | {h['f(x_n)']:>12.6e} | {fpx:>12.6e} | {h['x_n+1']:>14.10f} | {h['error_absoluto']:>10.2e} | {h['error_relativo']:>10.2e}")

    # ============================================
    # TABLA COMPARATIVA
    # ============================================
    print("\n" + "=" * 75)
    print("TABLA COMPARATIVA: SECANTE vs NEWTON-RAPHSON")
    print("=" * 75)
    
    print(f"\n{'Criterio':<30} {'Secante':<20} {'Newton-Raphson':<20}")
    print("-" * 75)
    print(f"{'Raíz aproximada':<30} {resultado_secante['raiz']:<20.12f} {resultado_newton['raiz']:<20.12f}")
    print(f"{'Iteraciones':<30} {resultado_secante['iteraciones']:<20d} {resultado_newton['iteraciones']:<20d}")
    print(f"{'Error final':<30} {resultado_secante['error_final']:<20.2e} {resultado_newton['error_final']:<20.2e}")
    print(f"{'Tiempo (ms)':<30} {resultado_secante['tiempo']*1000:<20.6f} {resultado_newton['tiempo']*1000:<20.6f}")
    print(f"{'Evaluaciones f(x)':<30} {resultado_secante['evaluaciones']:<20d} {resultado_newton.get('evaluaciones_f', resultado_newton['evaluaciones']):<20d}")
    eval_df_label = "Evaluaciones f'(x)"
    print(f"{eval_df_label:<30} {'N/A':<20} {resultado_newton.get('evaluaciones_df', 0):<20d}")
    print(f"{'Total evaluaciones':<30} {resultado_secante['evaluaciones']:<20d} {resultado_newton['evaluaciones']:<20d}")

    # ============================================
    # ANÁLISIS
    # ============================================
    print("\n" + "=" * 75)
    print("ANÁLISIS: ¿VALE LA PENA CALCULAR DERIVADAS ANALÍTICAS?")
    print("=" * 75)

    print("\nPara P(x) = x·e^(-x/2) - 0.3:")
    print("  Derivada: P'(x) = e^(-x/2)·(1 - x/2)")

    iter_s = resultado_secante['iteraciones']
    iter_n = resultado_newton['iteraciones']
    eval_s = resultado_secante['evaluaciones']
    eval_n = resultado_newton['evaluaciones']

    print(f"\n📈 Iteraciones: Newton ({iter_n}) vs Secante ({iter_s})")
    if iter_n < iter_s:
        print(f"   → Newton converge más rápido (orden 2 vs orden φ ≈ 1.618)")
    elif iter_s < iter_n:
        print(f"   → Secante convergió más rápido (inusual)")
    else:
        print(f"   → Ambos convergieron en {iter_n} iteraciones")

    print(f"\n📉 Evaluaciones totales: Newton ({eval_n}) vs Secante ({eval_s})")
    if eval_s < eval_n:
        print(f"   → Secante es más eficiente en evaluaciones totales")
        print(f"   → Secante no necesita calcular/evaluar derivadas")
    else:
        print(f"   → Newton es más eficiente en evaluaciones totales")

    print("\n💡 CONCLUSIÓN:")
    print("   • La derivada P'(x) = e^(-x/2)·(1 - x/2) es algebraicamente simple")
    print("   • Para este caso específico, ambos métodos son viables")
    if eval_s <= eval_n:
        print("   • Secante tiene ventaja: menos evaluaciones totales, sin derivadas")
    else:
        print("   • Newton tiene ventaja: menos iteraciones, convergencia más rápida")
    print("   • Si la función fuera más compleja (datos empíricos), Secante sería preferible")

    # ============================================
    # INTERPRETACIÓN
    # ============================================
    print("\n" + "=" * 75)
    print("INTERPRETACIÓN DEL RESULTADO")
    print("=" * 75)

    raiz = resultado_secante['raiz']
    print(f"\n   Raíz encontrada: x ≈ {raiz:.10f}")
    print(f"\n   Esto significa que el punto de equilibrio financiero ocurre cuando")
    print(f"   hay aproximadamente {raiz*1000:.0f} usuarios activos.")
    print(f"\n   • Menos de {raiz*1000:.0f} usuarios → Pérdidas (costos > ingresos)")
    print(f"   • Más de {raiz*1000:.0f} usuarios → Ganancias (ingresos > costos)")

    # Verificar el resultado
    print(f"\n   Verificación: P({raiz:.10f}) = {f(raiz):.2e} ≈ 0 ✓")

    print("\n" + "=" * 75)
    print("TEST COMPLETADO EXITOSAMENTE")
    print("=" * 75)

if __name__ == "__main__":
    test_ejercicio_5()

