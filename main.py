"""
Aplicación de Métodos Numéricos para Resolución de Ecuaciones No Lineales.

Este programa implementa una interfaz gráfica para resolver ecuaciones
no lineales utilizando cinco métodos numéricos diferentes:

1. Método de Bisección
2. Método de Falsa Posición (Regula Falsi)
3. Método de Punto Fijo
4. Método de Newton-Raphson
5. Método de la Secante

Autor: Proyecto de Análisis Numérico
Fecha: 2024
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interfaz.gui_principal import AplicacionMetodosNumericos
import tkinter as tk


def main():
    """
    Punto de entrada principal de la aplicación.

    Inicializa la ventana principal y ejecuta el loop de eventos.
    """
    # Crear ventana principal
    root = tk.Tk()

    # Configurar icono y propiedades
    root.title("Métodos Numéricos - Análisis Numérico")

    # Centrar ventana en pantalla
    ancho = 1400
    alto = 900
    x = (root.winfo_screenwidth() // 2) - (ancho // 2)
    y = (root.winfo_screenheight() // 2) - (alto // 2)
    root.geometry(f'{ancho}x{alto}+{x}+{y}')

    # Crear aplicación
    app = AplicacionMetodosNumericos(root)

    # Ejecutar loop principal
    root.mainloop()


if __name__ == "__main__":
    main()

