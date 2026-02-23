"""
Interfaz gráfica principal para la aplicación de métodos numéricos.

Este módulo contiene la ventana principal de la aplicación con todos
los componentes de la interfaz de usuario.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import warnings

# Suprimir warnings de glifos faltantes en matplotlib
warnings.filterwarnings('ignore', message='Glyph.*missing from.*font')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Dict, Any, Optional, Callable

# Importar módulos del proyecto
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metodos.biseccion import MetodoBiseccion
from metodos.falsa_posicion import MetodoFalsaPosicion
from metodos.punto_fijo import MetodoPuntoFijo
from metodos.newton import MetodoNewton
from metodos.secante import MetodoSecante
from funciones.definiciones import (
    FUNCIONES_EJERCICIOS,
    obtener_funcion,
    obtener_derivada,
    listar_ejercicios
)


class AplicacionMetodosNumericos:
    """
    Clase principal de la interfaz gráfica.

    Implementa una GUI completa con Tkinter para resolver ecuaciones
    no lineales usando diferentes métodos numéricos.
    """

    def __init__(self, root: tk.Tk):
        """
        Inicializa la aplicación.

        Args:
            root: Ventana principal de Tkinter.
        """
        self.root = root
        self.root.title("Métodos Numéricos - Resolución de Ecuaciones No Lineales")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Variables
        self.resultado_actual: Optional[Dict[str, Any]] = None
        self.funcion_actual: Optional[Callable] = None
        self.derivada_actual: Optional[Callable] = None

        # Configurar estilo
        self._configurar_estilo()

        # Crear interfaz
        self._crear_interfaz()

        # Cargar ejercicio por defecto
        self._cargar_ejercicio()

    def _configurar_estilo(self):
        """Configura el estilo de la aplicación con un diseño moderno."""
        style = ttk.Style()
        style.theme_use('clam')

        # Colores del tema
        COLOR_FONDO = '#f5f6fa'
        COLOR_PRIMARIO = '#2E86AB'
        COLOR_SECUNDARIO = '#A23B72'
        COLOR_ACENTO = '#F18F01'
        COLOR_TEXTO = '#2c3e50'
        COLOR_BORDE = '#dcdde1'

        # Configurar colores de fondo
        style.configure('TFrame', background=COLOR_FONDO)
        style.configure('TLabelframe', background=COLOR_FONDO)
        style.configure('TLabelframe.Label', background=COLOR_FONDO,
                       font=('Segoe UI', 11, 'bold'), foreground=COLOR_PRIMARIO)

        # Etiquetas
        style.configure('TLabel', background=COLOR_FONDO, font=('Segoe UI', 10),
                       foreground=COLOR_TEXTO)
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'),
                       foreground=COLOR_PRIMARIO, background=COLOR_FONDO)
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'),
                       foreground=COLOR_SECUNDARIO, background=COLOR_FONDO)
        style.configure('Result.TLabel', font=('Consolas', 10), background=COLOR_FONDO)

        # Botones con estilo moderno
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=(10, 5))
        style.map('TButton',
                  background=[('active', COLOR_PRIMARIO), ('!active', '#ffffff')],
                  foreground=[('active', 'white'), ('!active', COLOR_TEXTO)])

        # Botón primario (Calcular)
        style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'),
                       padding=(15, 8), background=COLOR_PRIMARIO)
        style.map('Primary.TButton',
                  background=[('active', '#1a5276'), ('!active', COLOR_PRIMARIO)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        # Combobox
        style.configure('TCombobox', font=('Segoe UI', 10), padding=5)

        # Entry
        style.configure('TEntry', font=('Segoe UI', 10), padding=5)

        # Notebook (pestañas)
        style.configure('TNotebook', background=COLOR_FONDO)
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'),
                       padding=(12, 6), background='#e8e8e8')
        style.map('TNotebook.Tab',
                  background=[('selected', COLOR_PRIMARIO), ('!selected', '#e8e8e8')],
                  foreground=[('selected', 'white'), ('!selected', COLOR_TEXTO)])

        # Estilo para Treeview mejorado
        style.configure('Treeview',
                       font=('Consolas', 9),
                       rowheight=28,
                       background='white',
                       fieldbackground='white')
        style.configure('Treeview.Heading',
                       font=('Segoe UI', 9, 'bold'),
                       background=COLOR_PRIMARIO,
                       foreground='white')
        style.map('Treeview',
                  background=[('selected', COLOR_PRIMARIO)],
                  foreground=[('selected', 'white')])

        # Scrollbar
        style.configure('TScrollbar', background=COLOR_BORDE, troughcolor=COLOR_FONDO)

        # Configurar el fondo de la ventana principal
        self.root.configure(bg=COLOR_FONDO)

    def _crear_interfaz(self):
        """Crea todos los componentes de la interfaz."""
        # Frame principal con PanedWindow para redimensionar
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo (entrada y resultados)
        self.frame_izquierdo = ttk.Frame(self.paned, width=500)
        self.paned.add(self.frame_izquierdo, weight=1)

        # Panel derecho (gráficas)
        self.frame_derecho = ttk.Frame(self.paned, width=700)
        self.paned.add(self.frame_derecho, weight=2)

        # Crear componentes
        self._crear_panel_entrada()
        self._crear_panel_resultados()
        self._crear_panel_tabla()
        self._crear_panel_graficas()

    def _crear_panel_entrada(self):
        """Crea el panel de entrada de datos."""
        # Frame de entrada
        frame_entrada = ttk.LabelFrame(
            self.frame_izquierdo,
            text="⚙️ Parámetros de Entrada",
            padding=10
        )
        frame_entrada.pack(fill=tk.X, padx=5, pady=5)

        # Selector de método
        ttk.Label(frame_entrada, text="Método:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5
        )

        self.metodo_var = tk.StringVar()
        self.combo_metodo = ttk.Combobox(
            frame_entrada,
            textvariable=self.metodo_var,
            values=[
                "Bisección",
                "Falsa Posición",
                "Punto Fijo",
                "Newton-Raphson",
                "Secante"
            ],
            state='readonly',
            width=25
        )
        self.combo_metodo.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)
        self.combo_metodo.set("Bisección")
        self.combo_metodo.bind('<<ComboboxSelected>>', self._on_metodo_cambio)

        # Selector de ejercicio
        ttk.Label(frame_entrada, text="Ejercicio:", style='Header.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=5
        )

        self.ejercicio_var = tk.StringVar()
        self.combo_ejercicio = ttk.Combobox(
            frame_entrada,
            textvariable=self.ejercicio_var,
            values=listar_ejercicios(),
            state='readonly',
            width=35
        )
        self.combo_ejercicio.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)
        self.combo_ejercicio.current(0)
        self.combo_ejercicio.bind('<<ComboboxSelected>>', self._cargar_ejercicio)

        # Función
        ttk.Label(frame_entrada, text="Función f(x):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.funcion_var = tk.StringVar()
        self.entry_funcion = ttk.Entry(
            frame_entrada,
            textvariable=self.funcion_var,
            width=40
        )
        self.entry_funcion.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=5)

        # Frame para intervalos/valores iniciales
        self.frame_valores = ttk.Frame(frame_entrada)
        self.frame_valores.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5)

        # Intervalo a
        ttk.Label(self.frame_valores, text="a:").grid(row=0, column=0, padx=5)
        self.a_var = tk.StringVar(value="0.5")
        self.entry_a = ttk.Entry(self.frame_valores, textvariable=self.a_var, width=10)
        self.entry_a.grid(row=0, column=1, padx=5)

        # Intervalo b
        ttk.Label(self.frame_valores, text="b:").grid(row=0, column=2, padx=5)
        self.b_var = tk.StringVar(value="2.5")
        self.entry_b = ttk.Entry(self.frame_valores, textvariable=self.b_var, width=10)
        self.entry_b.grid(row=0, column=3, padx=5)

        # Valor inicial x0
        self.label_x0 = ttk.Label(self.frame_valores, text="x₀:")
        self.label_x0.grid(row=0, column=4, padx=5)
        self.x0_var = tk.StringVar(value="1.0")
        self.entry_x0 = ttk.Entry(self.frame_valores, textvariable=self.x0_var, width=10)
        self.entry_x0.grid(row=0, column=5, padx=5)

        # Valor inicial x1 (para secante)
        self.label_x1 = ttk.Label(self.frame_valores, text="x₁:")
        self.label_x1.grid(row=0, column=6, padx=5)
        self.x1_var = tk.StringVar(value="1.0")
        self.entry_x1 = ttk.Entry(self.frame_valores, textvariable=self.x1_var, width=10)
        self.entry_x1.grid(row=0, column=7, padx=5)

        # Tolerancia
        ttk.Label(frame_entrada, text="Tolerancia:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.tol_var = tk.StringVar(value="1e-6")
        self.entry_tol = ttk.Entry(frame_entrada, textvariable=self.tol_var, width=15)
        self.entry_tol.grid(row=4, column=1, sticky=tk.W, pady=5)

        # Máximo de iteraciones
        ttk.Label(frame_entrada, text="Máx. iteraciones:").grid(
            row=5, column=0, sticky=tk.W, pady=5
        )
        self.max_iter_var = tk.StringVar(value="100")
        self.entry_max_iter = ttk.Entry(
            frame_entrada,
            textvariable=self.max_iter_var,
            width=15
        )
        self.entry_max_iter.grid(row=5, column=1, sticky=tk.W, pady=5)

        # Botones - Primera fila (principales)
        frame_botones = ttk.Frame(frame_entrada)
        frame_botones.grid(row=6, column=0, columnspan=3, pady=(15, 5))

        self.btn_calcular = ttk.Button(
            frame_botones,
            text="📊 Calcular",
            command=self._calcular,
            width=12,
            style='Primary.TButton'
        )
        self.btn_calcular.pack(side=tk.LEFT, padx=5)

        self.btn_limpiar = ttk.Button(
            frame_botones,
            text="🗑️ Limpiar",
            command=self._limpiar,
            width=12
        )
        self.btn_limpiar.pack(side=tk.LEFT, padx=5)

        self.btn_comparar = ttk.Button(
            frame_botones,
            text="📈 Bis vs FP",
            command=self._comparar_metodos,
            width=12
        )
        self.btn_comparar.pack(side=tk.LEFT, padx=5)

        # Botones - Segunda fila (comparaciones avanzadas)
        frame_botones2 = ttk.Frame(frame_entrada)
        frame_botones2.grid(row=7, column=0, columnspan=3, pady=(5, 10))

        self.btn_comparar_newton_secante = ttk.Button(
            frame_botones2,
            text="⚡ Newton vs Secante",
            command=self._comparar_newton_secante,
            width=18
        )
        self.btn_comparar_newton_secante.pack(side=tk.LEFT, padx=5)

        self.btn_comparar_x0 = ttk.Button(
            frame_botones2,
            text="🔄 Comparar Valores x₀",
            command=self._comparar_valores_iniciales,
            width=18
        )
        self.btn_comparar_x0.pack(side=tk.LEFT, padx=5)

        # Actualizar visibilidad de campos
        self._actualizar_campos_entrada()

    def _crear_panel_resultados(self):
        """Crea el panel de resultados finales."""
        frame_resultados = ttk.LabelFrame(
            self.frame_izquierdo,
            text="📋 Resultados",
            padding=10
        )
        frame_resultados.pack(fill=tk.X, padx=5, pady=5)

        # Texto de resultados con mejor estilo
        self.text_resultados = tk.Text(
            frame_resultados,
            height=9,
            font=('Consolas', 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg='#ffffff',
            fg='#2c3e50',
            relief='flat',
            borderwidth=2,
            padx=10,
            pady=8
        )
        self.text_resultados.pack(fill=tk.X, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(frame_resultados, command=self.text_resultados.yview)
        self.text_resultados.configure(yscrollcommand=scrollbar.set)

    def _crear_panel_tabla(self):
        """Crea el panel con la tabla de iteraciones."""
        frame_tabla = ttk.LabelFrame(
            self.frame_izquierdo,
            text="📊 Tabla de Iteraciones",
            padding=5
        )
        frame_tabla.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Crear Treeview con scrollbars
        self.tree_frame = ttk.Frame(frame_tabla)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        scroll_y = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)

        # Treeview
        self.tree = ttk.Treeview(
            self.tree_frame,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            show='headings'
        )

        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _crear_panel_graficas(self):
        """Crea el panel de gráficas con matplotlib."""
        # Notebook para múltiples gráficas
        self.notebook_graficas = ttk.Notebook(self.frame_derecho)
        self.notebook_graficas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Gráfica de la función
        self.frame_grafica_funcion = ttk.Frame(self.notebook_graficas)
        self.notebook_graficas.add(self.frame_grafica_funcion, text="Función")

        self.fig_funcion = Figure(figsize=(8, 6), dpi=100)
        self.ax_funcion = self.fig_funcion.add_subplot(111)
        self.canvas_funcion = FigureCanvasTkAgg(self.fig_funcion, self.frame_grafica_funcion)
        self.canvas_funcion.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar1 = NavigationToolbar2Tk(self.canvas_funcion, self.frame_grafica_funcion)
        toolbar1.update()

        # Tab 2: Gráfica de convergencia
        self.frame_grafica_convergencia = ttk.Frame(self.notebook_graficas)
        self.notebook_graficas.add(self.frame_grafica_convergencia, text="Convergencia")

        self.fig_convergencia = Figure(figsize=(8, 6), dpi=100)
        self.ax_convergencia = self.fig_convergencia.add_subplot(111)
        self.canvas_convergencia = FigureCanvasTkAgg(
            self.fig_convergencia,
            self.frame_grafica_convergencia
        )
        self.canvas_convergencia.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar2 = NavigationToolbar2Tk(
            self.canvas_convergencia,
            self.frame_grafica_convergencia
        )
        toolbar2.update()

        # Tab 3: Gráfica especial (cobweb, tangentes, secantes)
        self.frame_grafica_especial = ttk.Frame(self.notebook_graficas)
        self.notebook_graficas.add(self.frame_grafica_especial, text="Especial")

        self.fig_especial = Figure(figsize=(8, 6), dpi=100)
        self.ax_especial = self.fig_especial.add_subplot(111)
        self.canvas_especial = FigureCanvasTkAgg(
            self.fig_especial,
            self.frame_grafica_especial
        )
        self.canvas_especial.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar3 = NavigationToolbar2Tk(self.canvas_especial, self.frame_grafica_especial)
        toolbar3.update()

        # Tab 4: Comparación Newton vs Secante
        self.frame_grafica_comparacion = ttk.Frame(self.notebook_graficas)
        self.notebook_graficas.add(self.frame_grafica_comparacion, text="Newton vs Secante")

        self.fig_comparacion = Figure(figsize=(8, 6), dpi=100)
        self.ax_comparacion = self.fig_comparacion.add_subplot(111)
        self.canvas_comparacion = FigureCanvasTkAgg(
            self.fig_comparacion,
            self.frame_grafica_comparacion
        )
        self.canvas_comparacion.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar4 = NavigationToolbar2Tk(self.canvas_comparacion, self.frame_grafica_comparacion)
        toolbar4.update()

    def _on_metodo_cambio(self, event=None):
        """Maneja el cambio de método seleccionado."""
        self._actualizar_campos_entrada()

        # Actualizar combo de ejercicio según método
        metodo = self.metodo_var.get()
        ejercicios = listar_ejercicios()

        ejercicio_map = {
            "Bisección": 0,
            "Falsa Posición": 1,
            "Punto Fijo": 2,
            "Newton-Raphson": 3,
            "Secante": 4
        }

        if metodo in ejercicio_map:
            self.combo_ejercicio.current(ejercicio_map[metodo])
            self._cargar_ejercicio()

    def _actualizar_campos_entrada(self):
        """Actualiza la visibilidad de los campos según el método."""
        metodo = self.metodo_var.get()

        # Ocultar todos primero
        for widget in [self.entry_a, self.entry_b, self.entry_x0, self.entry_x1,
                       self.label_x0, self.label_x1]:
            widget.grid_remove()

        for widget in self.frame_valores.winfo_children():
            if isinstance(widget, ttk.Label):
                widget.grid_remove()

        if metodo in ["Bisección", "Falsa Posición"]:
            # Mostrar a y b
            ttk.Label(self.frame_valores, text="a:").grid(row=0, column=0, padx=5)
            self.entry_a.grid(row=0, column=1, padx=5)
            ttk.Label(self.frame_valores, text="b:").grid(row=0, column=2, padx=5)
            self.entry_b.grid(row=0, column=3, padx=5)

        elif metodo in ["Punto Fijo", "Newton-Raphson"]:
            # Mostrar x0
            ttk.Label(self.frame_valores, text="x₀:").grid(row=0, column=0, padx=5)
            self.entry_x0.grid(row=0, column=1, padx=5)

        elif metodo == "Secante":
            # Mostrar x0 y x1
            ttk.Label(self.frame_valores, text="x₀:").grid(row=0, column=0, padx=5)
            self.entry_x0.grid(row=0, column=1, padx=5)
            ttk.Label(self.frame_valores, text="x₁:").grid(row=0, column=2, padx=5)
            self.entry_x1.grid(row=0, column=3, padx=5)

    def _cargar_ejercicio(self, event=None):
        """Carga los parámetros del ejercicio seleccionado."""
        ejercicio_nombre = self.ejercicio_var.get()

        if not ejercicio_nombre:
            return

        try:
            ejercicio = FUNCIONES_EJERCICIOS[ejercicio_nombre]
        except KeyError:
            return

        # Cargar función
        if 'funcion_original' in ejercicio:
            # Para punto fijo, mostramos g(x)
            self.funcion_var.set(ejercicio['funcion_str'])
        else:
            self.funcion_var.set(ejercicio['funcion_str'])

        # Cargar parámetros según el tipo
        if 'intervalo' in ejercicio:
            a, b = ejercicio['intervalo']
            self.a_var.set(str(a))
            self.b_var.set(str(b))

        if 'x0' in ejercicio:
            self.x0_var.set(str(ejercicio['x0']))

        if 'x1' in ejercicio:
            self.x1_var.set(str(ejercicio['x1']))

        self.tol_var.set(str(ejercicio['tolerancia']))
        self.max_iter_var.set(str(ejercicio['max_iter']))

        # Actualizar método
        metodo_map = {
            'biseccion': 'Bisección',
            'falsa_posicion': 'Falsa Posición',
            'punto_fijo': 'Punto Fijo',
            'newton': 'Newton-Raphson',
            'secante': 'Secante'
        }
        self.metodo_var.set(metodo_map.get(ejercicio['metodo'], 'Bisección'))
        self._actualizar_campos_entrada()

    def _calcular(self):
        """Ejecuta el método numérico seleccionado."""
        metodo = self.metodo_var.get()

        try:
            # Obtener parámetros comunes
            tolerancia = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())
            funcion_str = self.funcion_var.get()

            # Obtener función
            f = obtener_funcion(funcion_str)
            self.funcion_actual = f

            # Ejecutar según método
            if metodo == "Bisección":
                resultado = self._ejecutar_biseccion(f, tolerancia, max_iter)
            elif metodo == "Falsa Posición":
                resultado = self._ejecutar_falsa_posicion(f, tolerancia, max_iter)
            elif metodo == "Punto Fijo":
                resultado = self._ejecutar_punto_fijo(f, tolerancia, max_iter)
            elif metodo == "Newton-Raphson":
                resultado = self._ejecutar_newton(f, funcion_str, tolerancia, max_iter)
            elif metodo == "Secante":
                resultado = self._ejecutar_secante(f, tolerancia, max_iter)
            else:
                messagebox.showerror("Error", f"Método no reconocido: {metodo}")
                return

            self.resultado_actual = resultado

            # Mostrar resultados
            self._mostrar_resultados(resultado)
            self._actualizar_tabla(resultado)
            self._graficar_funcion(resultado)
            self._graficar_convergencia(resultado)
            self._graficar_especial(resultado, metodo)

            # Para el método Secante, generar automáticamente la comparación con Newton
            if metodo == "Secante":
                self._graficar_comparacion_automatica_secante(resultado, funcion_str, tolerancia, max_iter)

        except ValueError as e:
            messagebox.showerror("Error de Valor", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")

    def _ejecutar_biseccion(
        self,
        f: Callable,
        tolerancia: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Ejecuta el método de bisección."""
        a = float(self.a_var.get())
        b = float(self.b_var.get())

        metodo = MetodoBiseccion(f, a, b, tolerancia, max_iter)
        return metodo.resolver()

    def _ejecutar_falsa_posicion(
        self,
        f: Callable,
        tolerancia: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Ejecuta el método de falsa posición."""
        a = float(self.a_var.get())
        b = float(self.b_var.get())

        metodo = MetodoFalsaPosicion(f, a, b, tolerancia, max_iter)
        return metodo.resolver()

    def _ejecutar_punto_fijo(
        self,
        g: Callable,
        tolerancia: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Ejecuta el método de punto fijo."""
        x0 = float(self.x0_var.get())

        # Obtener derivada de g para verificar convergencia
        funcion_str = self.funcion_var.get()
        try:
            dg = obtener_derivada(funcion_str)
        except:
            dg = None

        metodo = MetodoPuntoFijo(g, x0, tolerancia, max_iter, dg)
        return metodo.resolver()

    def _ejecutar_newton(
        self,
        f: Callable,
        funcion_str: str,
        tolerancia: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Ejecuta el método de Newton-Raphson."""
        x0 = float(self.x0_var.get())

        # Obtener derivada
        df = obtener_derivada(funcion_str)
        self.derivada_actual = df

        metodo = MetodoNewton(f, df, x0, tolerancia, max_iter)
        return metodo.resolver()

    def _ejecutar_secante(
        self,
        f: Callable,
        tolerancia: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Ejecuta el método de la secante."""
        x0 = float(self.x0_var.get())
        x1 = float(self.x1_var.get())

        metodo = MetodoSecante(f, x0, x1, tolerancia, max_iter)
        return metodo.resolver()

    def _mostrar_resultados(self, resultado: Dict[str, Any]):
        """
        Muestra los resultados en el panel de texto con formato mejorado.

        Incluye información clara sobre convergencia/falla y estadísticas detalladas.
        """
        self.text_resultados.config(state=tk.NORMAL)
        self.text_resultados.delete(1.0, tk.END)

        # Obtener valores con manejo de None
        raiz = resultado.get('raiz')
        error_final = resultado.get('error_final')
        convergencia = resultado.get('convergencia', False)
        metodo = resultado.get('metodo', 'Método')

        raiz_str = f"{raiz:.12f}" if raiz is not None else "N/A"
        error_str = f"{error_final:.8e}" if error_final is not None and error_final != float('inf') else "N/A"

        # Calcular error relativo en porcentaje
        error_rel_pct = "N/A"
        if raiz is not None and raiz != 0 and error_final is not None and error_final != float('inf'):
            error_rel_pct = f"{(error_final / abs(raiz)) * 100:.8e} %"

        # Símbolo de estado
        if convergencia:
            estado = "✅ CONVERGENCIA EXITOSA"
            estado_detalle = "El método encontró una raíz dentro de la tolerancia especificada."
        else:
            estado = "❌ NO CONVERGIÓ"
            if resultado.get('divergencia'):
                estado_detalle = "El método divergió. Pruebe con otro valor inicial."
            else:
                estado_detalle = "Se alcanzó el máximo de iteraciones sin converger."

        # Formatear resultados
        texto = f"""{'═'*55}
   RESULTADOS - {metodo}
{'═'*55}

{estado}
{estado_detalle}

{'─'*55}
📍 Raíz aproximada:     {raiz_str}
📊 Iteraciones:         {resultado.get('iteraciones', 'N/A')}
⚡ Error absoluto:       {error_str}
📐 Error relativo:      {error_rel_pct}
⏱️  Tiempo ejecución:    {resultado.get('tiempo', 0)*1000:.6f} ms
📈 Evaluaciones f(x):   {resultado.get('evaluaciones', resultado.get('evaluaciones_f', 'N/A'))}
{'─'*55}

📝 {resultado.get('mensaje', '')}
"""

        # Agregar información adicional según el método
        if metodo == 'Newton-Raphson':
            texto += f"""
╔══════════════════════════════════════════════════════╗
║          INFORMACIÓN ADICIONAL - NEWTON              ║
╠══════════════════════════════════════════════════════╣
║ Evaluaciones de f(x):  {resultado.get('evaluaciones_f', 'N/A'):<27} ║
║ Evaluaciones de f'(x): {resultado.get('evaluaciones_df', 'N/A'):<27} ║
║ Orden de convergencia: Cuadrático (≈2)              ║
╚══════════════════════════════════════════════════════╝
"""
        elif metodo == 'Secante':
            texto += f"""
╔══════════════════════════════════════════════════════╗
║          INFORMACIÓN ADICIONAL - SECANTE             ║
╠══════════════════════════════════════════════════════╣
║ Evaluaciones de f(x):  {resultado.get('evaluaciones', 'N/A'):<27} ║
║ No requiere derivadas                                ║
║ Orden de convergencia: Superlineal (≈1.618)         ║
╚══════════════════════════════════════════════════════╝
"""
        elif metodo == 'Punto Fijo':
            condicion = resultado.get('condicion_g_prima')
            if condicion is not None:
                cumple = "✓ SÍ" if condicion < 1 else "✗ NO"
                texto += f"""
╔══════════════════════════════════════════════════════╗
║          CONDICIÓN DE CONVERGENCIA                   ║
╠══════════════════════════════════════════════════════╣
║ |g'(x)| = {condicion:<10.6f}                            ║
║ ¿Cumple |g'(x)| < 1? {cumple:<31} ║
╚══════════════════════════════════════════════════════╝
"""

        # Agregar interpretación del resultado si está disponible
        ejercicio_nombre = self.ejercicio_var.get()
        if ejercicio_nombre and ejercicio_nombre in FUNCIONES_EJERCICIOS:
            ejercicio_info = FUNCIONES_EJERCICIOS[ejercicio_nombre]
            if 'interpretacion' in ejercicio_info and convergencia:
                texto += f"""
╔══════════════════════════════════════════════════════╗
║          INTERPRETACIÓN DEL RESULTADO                ║
╠══════════════════════════════════════════════════════╣
{self._formatear_texto_caja(ejercicio_info['interpretacion'], 54)}
╚══════════════════════════════════════════════════════╝
"""
            if 'descripcion' in ejercicio_info:
                texto += f"""
📋 Problema: {ejercicio_info['descripcion']}
"""
            # Mostrar valor de la función original si existe
            if 'funcion_original' in ejercicio_info and raiz is not None:
                try:
                    f_original = obtener_funcion(ejercicio_info['funcion_original'])
                    valor_T = f_original(raiz)
                    texto += f"""
🎯 Valor de T(λ={raiz:.6f}) = {valor_T:.8f}
   Este es el tiempo mínimo de búsqueda en la hash table.
"""
                except:
                    pass

        self.text_resultados.insert(tk.END, texto)
        self.text_resultados.config(state=tk.DISABLED)

    def _formatear_texto_caja(self, texto: str, ancho: int) -> str:
        """Formatea texto para que quepa en una caja de ancho fijo."""
        import textwrap
        lineas = textwrap.wrap(texto, width=ancho - 4)
        resultado = ""
        for linea in lineas:
            resultado += f"║ {linea:<{ancho-2}} ║\n"
        return resultado.rstrip('\n')

    def _actualizar_tabla(self, resultado: Dict[str, Any]):
        """Actualiza la tabla de iteraciones con formato de 8 decimales."""
        # Limpiar tabla anterior
        for item in self.tree.get_children():
            self.tree.delete(item)

        historial = resultado.get('historial', [])
        if not historial:
            return

        # Obtener columnas del primer elemento
        columnas_originales = list(historial[0].keys())

        # Renombrar error_relativo para mostrar porcentaje
        columnas = []
        for col in columnas_originales:
            if col == 'error_relativo':
                columnas.append('error_relativo (%)')
            else:
                columnas.append(col)

        # Configurar columnas
        self.tree['columns'] = columnas

        for col in columnas:
            self.tree.heading(col, text=col)
            # Ajustar ancho según tipo de columna
            if col == 'n':
                self.tree.column(col, width=50, anchor=tk.CENTER)
            elif 'error' in col.lower():
                self.tree.column(col, width=130, anchor=tk.CENTER)
            else:
                self.tree.column(col, width=120, anchor=tk.CENTER)

        # Insertar datos con formato mejorado
        for fila in historial:
            valores = []
            for col_orig in columnas_originales:
                valor = fila.get(col_orig, '')
                if isinstance(valor, float):
                    if col_orig == 'error_relativo':
                        # Convertir a porcentaje
                        porcentaje = valor * 100
                        if abs(porcentaje) < 1e-4 or abs(porcentaje) > 1e4:
                            valores.append(f"{porcentaje:.6e} %")
                        else:
                            valores.append(f"{porcentaje:.8f} %")
                    elif 'error' in col_orig.lower():
                        # Errores en notación científica
                        valores.append(f"{valor:.8e}")
                    elif abs(valor) < 1e-6 or abs(valor) > 1e6:
                        # Valores muy pequeños o grandes
                        valores.append(f"{valor:.8e}")
                    else:
                        # Valores normales con 8 decimales
                        valores.append(f"{valor:.8f}")
                else:
                    valores.append(str(valor))

            self.tree.insert('', tk.END, values=valores)

    def _graficar_funcion(self, resultado: Dict[str, Any]):
        """Grafica la función y la raíz encontrada."""
        self.ax_funcion.clear()

        if not self.funcion_actual:
            return

        raiz = resultado.get('raiz')
        metodo = resultado.get('metodo', '')

        # Configurar fondo
        self.ax_funcion.set_facecolor('#fafafa')

        # Determinar rango de gráfica
        if metodo in ['Bisección', 'Falsa Posición']:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            margen = (b - a) * 0.2
            x_min, x_max = a - margen, b + margen
        elif metodo == 'Secante':
            # Rango específico para Secante (Ejercicio 5)
            x_min, x_max = -0.2, 2.0
        elif raiz is not None:
            x_min, x_max = raiz - 2, raiz + 2
        else:
            x_min, x_max = -5, 5

        # Generar puntos para la gráfica
        x = np.linspace(x_min, x_max, 500)

        try:
            y = [self.funcion_actual(xi) for xi in x]
            y = np.array(y)

            # Limitar valores extremos para mejor visualización
            if metodo == 'Secante':
                y_clipped = np.clip(y, -0.6, 0.6)
            else:
                y_clipped = np.clip(y, -100, 100)
        except:
            return

        # Graficar función con mejor estilo
        self.ax_funcion.plot(x, y_clipped, color='#2c3e50', linewidth=3, label='f(x)')
        self.ax_funcion.fill_between(x, y_clipped, 0, alpha=0.15, color='#3498db')

        # Línea y=0 más visible
        self.ax_funcion.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=1.5)

        # Marcar raíz con estrella grande
        if raiz is not None:
            self.ax_funcion.plot(raiz, 0, '*', color='#27ae60', markersize=22,
                                markeredgecolor='white', markeredgewidth=2,
                                label=f'Raíz ≈ {raiz:.6f}', zorder=10)
            self.ax_funcion.axvline(x=raiz, color='#27ae60', linestyle=':', alpha=0.7, linewidth=2)

        # Marcar aproximaciones con colores degradados
        historial = resultado.get('historial', [])
        if historial:
            if 'c' in historial[0]:
                aproximaciones = [h['c'] for h in historial]
            elif 'x_n' in historial[0]:
                aproximaciones = [h['x_n'] for h in historial]
            elif 'x_{n+1}' in historial[0]:
                aproximaciones = [h['x_{n+1}'] for h in historial]
            else:
                aproximaciones = []

            if aproximaciones:
                y_aprox = [self.funcion_actual(xi) for xi in aproximaciones]
                # Colores degradados del naranja al rojo
                colores = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(aproximaciones)))
                for i, (xa, ya) in enumerate(zip(aproximaciones, y_aprox)):
                    self.ax_funcion.scatter(xa, ya, c=[colores[i]], s=80,
                                           edgecolors='white', linewidths=1.5,
                                           zorder=5, alpha=0.9)
                # Añadir leyenda para aproximaciones
                self.ax_funcion.scatter([], [], c='#e74c3c', s=80,
                                       edgecolors='white', linewidths=1.5,
                                       label='Aproximaciones')

        self.ax_funcion.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_funcion.set_ylabel('f(x)', fontsize=13, fontweight='bold')
        self.ax_funcion.set_title(f'Gráfica de la Función - {metodo}', fontsize=14, fontweight='bold', pad=10)
        self.ax_funcion.legend(loc='best', fontsize=10, framealpha=0.95, fancybox=True)
        self.ax_funcion.grid(True, alpha=0.4, linestyle='--')

        # Ajustar límites Y para Secante
        if metodo == 'Secante':
            self.ax_funcion.set_ylim(-0.5, 0.5)

        self.fig_funcion.tight_layout()
        self.canvas_funcion.draw()

    def _graficar_convergencia(self, resultado: Dict[str, Any]):
        """Grafica la convergencia del error en escala logarítmica con análisis."""
        self.ax_convergencia.clear()

        historial = resultado.get('historial', [])
        if not historial:
            return

        iteraciones = [h['n'] for h in historial]

        # Obtener errores
        if 'error_absoluto' in historial[0]:
            errores = [h['error_absoluto'] for h in historial]
        else:
            return

        # Filtrar errores positivos para escala logarítmica
        errores_validos = [(i, e) for i, e in zip(iteraciones, errores) if e > 0]
        if not errores_validos:
            return

        iter_validas, err_validos = zip(*errores_validos)

        # Configurar fondo
        self.ax_convergencia.set_facecolor('#fafafa')

        # Gráfica principal en escala logarítmica con mejor estilo
        self.ax_convergencia.semilogy(
            iter_validas, err_validos, '-o',
            color='#2E86AB', linewidth=3, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5,
            label='Error Absoluto'
        )

        # Línea de tolerancia más visible
        tolerancia = float(self.tol_var.get())
        self.ax_convergencia.axhline(
            y=tolerancia,
            color='#E94F37',
            linestyle='--',
            linewidth=3,
            label=f'Tolerancia = {tolerancia:.0e}'
        )

        # Sombrear zona de convergencia
        self.ax_convergencia.axhspan(0, tolerancia, alpha=0.15, color='#27ae60', label='Zona convergida')

        # Análisis del orden de convergencia para Newton-Raphson
        metodo = resultado.get('metodo', '')
        if metodo == 'Newton-Raphson' and len(err_validos) >= 3:
            # Verificar convergencia cuadrática: e_{n+1}/e_n^2 ≈ constante
            ratios = []
            for i in range(1, len(err_validos)):
                if err_validos[i-1] > 1e-15:
                    ratio = err_validos[i] / (err_validos[i-1] ** 2)
                    ratios.append(ratio)

            if ratios:
                promedio_ratio = np.mean(ratios[-3:]) if len(ratios) >= 3 else np.mean(ratios)
                # Agregar texto de análisis con mejor estilo
                self.ax_convergencia.text(
                    0.05, 0.95,
                    f'Convergencia cuadrática\ne(n+1)/e(n)² = {promedio_ratio:.4f}',
                    transform=self.ax_convergencia.transAxes,
                    fontsize=11, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                             edgecolor='#2E86AB', linewidth=2, alpha=0.95)
                )

        # Agregar error relativo como segunda serie si hay suficientes datos
        if 'error_relativo' in historial[0]:
            errores_rel = [h['error_relativo'] for h in historial if h['error_relativo'] > 0]
            if len(errores_rel) == len(iter_validas):
                self.ax_convergencia.semilogy(
                    iter_validas, errores_rel, '-s',
                    color='#27ae60', linewidth=2, markersize=8,
                    markerfacecolor='white', markeredgewidth=2,
                    alpha=0.8, label='Error Relativo'
                )

        self.ax_convergencia.set_xlabel('Iteración', fontsize=13, fontweight='bold')
        self.ax_convergencia.set_ylabel('Error (escala logarítmica)', fontsize=13, fontweight='bold')
        self.ax_convergencia.set_title(
            f"Convergencia del Error - {metodo}",
            fontsize=14, fontweight='bold', pad=10
        )
        self.ax_convergencia.grid(True, alpha=0.4, which='both', linestyle='--')
        self.ax_convergencia.legend(loc='upper right', fontsize=10, framealpha=0.95, fancybox=True)

        # Agregar anotación del error final con mejor estilo
        if err_validos:
            error_final = err_validos[-1]
            self.ax_convergencia.annotate(
                f'Error final:\n{error_final:.2e}',
                xy=(iter_validas[-1], error_final),
                xytext=(iter_validas[-1]-0.8, error_final*50),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f39c12', alpha=0.9),
                ha='center'
            )

        self.fig_convergencia.tight_layout()
        self.canvas_convergencia.draw()

    def _graficar_especial(self, resultado: Dict[str, Any], metodo: str):
        """Genera gráficas especiales según el método."""
        self.ax_especial.clear()

        if metodo == "Punto Fijo":
            self._graficar_cobweb(resultado)
        elif metodo == "Newton-Raphson":
            self._graficar_tangentes(resultado)
        elif metodo == "Secante":
            self._graficar_secantes(resultado)
        else:
            # Para bisección y falsa posición, mostrar intervalos
            self._graficar_intervalos(resultado)

        self.fig_especial.tight_layout()
        self.canvas_especial.draw()

    def _graficar_cobweb(self, resultado: Dict[str, Any]):
        """
        Genera el cobweb plot completo para punto fijo.

        El diagrama de telaraña muestra:
        - y = x (línea de identidad)
        - y = g(x) (función de iteración)
        - Trayectoria de iteraciones formando la "telaraña"
        """
        historial = resultado.get('historial', [])
        if not historial:
            return

        # Configurar fondo
        self.ax_especial.set_facecolor('#f8f9fa')

        # Obtener rango para las gráficas
        x_valores = [h['x_n'] for h in historial] + [h['g(x_n)'] for h in historial]
        x_min = min(x_valores) - 0.5
        x_max = max(x_valores) + 0.5

        # Extender el rango para mejor visualización
        rango = x_max - x_min
        x_min -= rango * 0.1
        x_max += rango * 0.1

        x = np.linspace(x_min, x_max, 300)

        # Graficar y = x (línea de identidad) con estilo mejorado
        self.ax_especial.plot(x, x, color='#3498db', linewidth=3, label='y = x', zorder=1)
        self.ax_especial.fill_between(x, x, x_min, alpha=0.05, color='#3498db')

        # Graficar y = g(x)
        if self.funcion_actual:
            try:
                y = [self.funcion_actual(xi) for xi in x]
                self.ax_especial.plot(x, y, color='#e74c3c', linewidth=3, label='y = g(x)', zorder=2)
            except:
                pass

        # Dibujar el cobweb completo
        x0 = float(self.x0_var.get())

        # Colores degradados para las iteraciones (de naranja a verde)
        n_iter = len(historial)
        colores_cobweb = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60', '#1abc9c', '#16a085']

        # Primera línea vertical desde (x0, 0) hasta (x0, g(x0))
        if historial:
            primera_g = historial[0]['g(x_n)']
            self.ax_especial.plot([x0, x0], [0, primera_g], '--', color='#9b59b6', linewidth=2, alpha=0.7)

        # Dibujar cada iteración del cobweb
        for i, h in enumerate(historial):
            xn = h['x_n']
            gxn = h['g(x_n)']
            color = colores_cobweb[min(i, len(colores_cobweb)-1)]

            # Línea vertical: de (xn, xn) a (xn, g(xn))
            self.ax_especial.plot([xn, xn], [xn, gxn], '-', color=color, linewidth=2.5, alpha=0.9)

            # Línea horizontal: de (xn, g(xn)) a (g(xn), g(xn))
            self.ax_especial.plot([xn, gxn], [gxn, gxn], '-', color=color, linewidth=2.5, alpha=0.9)

            # Marcar el punto actual con círculo
            self.ax_especial.plot(xn, gxn, 'o', color=color, markersize=10,
                                 markeredgecolor='white', markeredgewidth=2, alpha=0.9, zorder=5)

        # Marcar punto inicial con estilo destacado
        self.ax_especial.plot(x0, 0, 'o', color='#2c3e50', markersize=14,
                             markeredgecolor='white', markeredgewidth=2,
                             label=f'x₀ = {x0:.2f}', zorder=6)

        # Marcar punto fijo encontrado con estrella grande
        raiz = resultado.get('raiz')
        if raiz:
            self.ax_especial.plot(raiz, raiz, '*', color='#27ae60', markersize=25,
                                   markeredgecolor='white', markeredgewidth=2,
                                   label=f'Punto fijo = {raiz:.6f}', zorder=10)
            # Dibujar líneas auxiliares al punto fijo
            self.ax_especial.axhline(y=raiz, color='#27ae60', linestyle=':', alpha=0.6, linewidth=2)
            self.ax_especial.axvline(x=raiz, color='#27ae60', linestyle=':', alpha=0.6, linewidth=2)

        # Agregar información sobre condición de convergencia con mejor estilo
        condicion = resultado.get('condicion_g_prima')
        if condicion is not None:
            texto_conv = f"|g'(x)| = {condicion:.4f}\n"
            texto_conv += "[OK] Converge" if condicion < 1 else "[!] Puede diverger"
            self.ax_especial.text(
                0.02, 0.98, texto_conv,
                transform=self.ax_especial.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                         edgecolor='#27ae60' if condicion < 1 else '#e74c3c',
                         linewidth=2, alpha=0.95)
            )

        self.ax_especial.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_especial.set_ylabel('y', fontsize=13, fontweight='bold')
        self.ax_especial.set_title(
            f'Diagrama de Telaraña (Cobweb) - Punto Fijo\n{len(historial)} iteraciones hasta convergencia',
            fontsize=14, fontweight='bold', pad=15
        )
        self.ax_especial.legend(loc='upper left', fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
        self.ax_especial.grid(True, alpha=0.4, linestyle='--')
        self.ax_especial.set_aspect('equal', adjustable='box')

    def _graficar_tangentes(self, resultado: Dict[str, Any]):
        """
        Grafica las tangentes dinámicamente para Newton-Raphson.

        Muestra la función f(x) y las rectas tangentes en cada iteración,
        ilustrando cómo el método converge hacia la raíz.
        """
        tangentes = resultado.get('tangentes', [])
        if not tangentes or not self.funcion_actual:
            return

        # Configurar fondo
        self.ax_especial.set_facecolor('#f8f9fa')

        # Obtener rango centrado en la raíz
        raiz = resultado.get('raiz')
        if raiz:
            x_min = raiz - 2.5
            x_max = raiz + 2.5
        else:
            x_valores = [t['x'] for t in tangentes] + [t['x_siguiente'] for t in tangentes]
            x_min = min(x_valores) - 1
            x_max = max(x_valores) + 1

        x = np.linspace(x_min, x_max, 500)

        # Graficar función con mejor estilo
        try:
            y = [self.funcion_actual(xi) for xi in x]
            y_array = np.array(y)
            # Limitar valores extremos
            y_clipped = np.clip(y_array, -30, 30)
            self.ax_especial.plot(x, y_clipped, color='#1a1a2e', linewidth=3.5, label='f(x) = x³ - 8x² + 20x - 16')
            self.ax_especial.fill_between(x, y_clipped, 0, alpha=0.08, color='#3498db')
        except:
            return

        # Línea y=0 más visible
        self.ax_especial.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=2)

        # Colores vibrantes para las tangentes (rojo -> naranja -> amarillo)
        colores_tang = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#27ae60', '#3498db', '#9b59b6', '#1abc9c']

        n_tangentes = min(len(tangentes), 8)

        for i, t in enumerate(tangentes[:n_tangentes]):
            x_t = t['x']
            fx = t['f(x)']
            dfx = t["f'(x)"]
            x_sig = t['x_siguiente']
            color = colores_tang[i]

            # Calcular el rango de la tangente
            x_tang_min = min(x_t, x_sig) - 0.8
            x_tang_max = max(x_t, x_sig) + 0.8
            x_tang = np.linspace(x_tang_min, x_tang_max, 100)

            # Ecuación de la tangente: y = fx + dfx*(x - x_t)
            y_tang = fx + dfx * (x_tang - x_t)
            y_tang = np.clip(y_tang, -35, 35)

            # Dibujar la tangente con línea sólida
            self.ax_especial.plot(
                x_tang, y_tang,
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.9,
                label=f'Tangente {i+1}: x={x_t:.2f}' if i < 5 else None
            )

            # Marcar el punto (x_n, f(x_n)) con círculo grande
            self.ax_especial.plot(x_t, fx, 'o', color=color, markersize=14,
                                 markeredgecolor='white', markeredgewidth=2, zorder=6)

            # Dibujar línea vertical punteada desde el punto hasta el eje x
            self.ax_especial.plot([x_t, x_t], [0, fx], '--', color=color, linewidth=1.5, alpha=0.6)

            # Marcar donde la tangente cruza el eje x (x_{n+1}) con cuadrado
            self.ax_especial.plot(x_sig, 0, 's', color=color, markersize=10,
                                 markeredgecolor='white', markeredgewidth=1.5, alpha=0.9, zorder=5)

            # Añadir flecha indicando la dirección de convergencia
            if i < len(tangentes) - 1:
                dx = x_sig - x_t
                if abs(dx) > 0.1:
                    self.ax_especial.annotate('', xy=(x_sig, 0.5), xytext=(x_t, 0.5),
                                             arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.5))

        # Marcar raíz final con estrella grande
        if raiz:
            self.ax_especial.plot(raiz, 0, '*', color='#27ae60', markersize=28,
                                   markeredgecolor='white', markeredgewidth=2,
                                   label=f'Raíz = {raiz:.6f}', zorder=10)
            # Línea vertical en la raíz
            self.ax_especial.axvline(x=raiz, color='#27ae60', linestyle=':', linewidth=2.5, alpha=0.7)

        # Marcar punto inicial con círculo negro grande
        x0 = float(self.x0_var.get())
        try:
            fx0 = self.funcion_actual(x0)
            self.ax_especial.plot(x0, fx0, 'o', color='#2c3e50', markersize=16,
                                   markeredgecolor='white', markeredgewidth=2,
                                   label=f'x₀ = {x0:.1f}', zorder=7)
        except:
            pass

        # Configuración del gráfico mejorada
        self.ax_especial.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_especial.set_ylabel('f(x)', fontsize=13, fontweight='bold')
        self.ax_especial.set_title(
            f'Método Newton-Raphson - Convergencia por Tangentes\n'
            f'{resultado.get("iteraciones", 0)} iteraciones hasta convergencia',
            fontsize=14, fontweight='bold', pad=15
        )
        self.ax_especial.legend(loc='upper right', fontsize=9, framealpha=0.95,
                               fancybox=True, shadow=True, ncol=2)
        self.ax_especial.grid(True, alpha=0.4, linestyle='--')

        # Ajustar límites del eje y con mejor escala
        if raiz:
            # Centrar en la raíz para mejor visualización
            self.ax_especial.set_xlim(raiz - 2, raiz + 2)
            self.ax_especial.set_ylim(-15, 25)

    def _graficar_secantes(self, resultado: Dict[str, Any]):
        """
        Grafica las rectas secantes dinámicamente.

        Muestra la función f(x) y las rectas secantes entre puntos consecutivos,
        ilustrando cómo el método converge hacia la raíz.
        """
        secantes = resultado.get('secantes', [])
        if not secantes or not self.funcion_actual:
            return

        # Configurar fondo
        self.ax_especial.set_facecolor('#fafafa')

        # Obtener rango
        x_valores = []
        for s in secantes:
            x_valores.extend([s['x_prev'], s['x_curr'], s['x_next']])

        x_min = min(x_valores) - 0.5
        x_max = max(x_valores) + 0.5

        # Incluir raíz en el rango
        raiz = resultado.get('raiz')
        if raiz:
            x_min = min(x_min, raiz - 0.3)
            x_max = max(x_max, raiz + 0.3)

        x = np.linspace(x_min, x_max, 400)

        # Graficar función con mejor estilo
        try:
            y = [self.funcion_actual(xi) for xi in x]
            y_array = np.array(y)
            y_clipped = np.clip(y_array, -1, 1)
            self.ax_especial.plot(x, y_clipped, color='#1a1a2e', linewidth=3, label='P(x) = x·e^(-x/2) - 0.3')
            self.ax_especial.fill_between(x, y_clipped, 0, alpha=0.1, color='#3498db')
        except:
            return

        # Línea y=0 más visible
        self.ax_especial.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=1.5)

        # Colores vibrantes para las secantes
        colores_sec = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c', '#e91e63']

        # Graficar todas las secantes
        n_secantes = min(len(secantes), 8)

        for i, s in enumerate(secantes[:n_secantes]):
            x1, x2 = s['x_prev'], s['x_curr']
            f1, f2 = s['f_prev'], s['f_curr']
            x_next = s['x_next']
            color = colores_sec[i]

            # Calcular la recta secante
            if abs(x2 - x1) > 1e-10:
                pendiente = (f2 - f1) / (x2 - x1)

                # Extender la secante para que sea visible
                x_sec_min = min(x1, x2, x_next) - 0.3
                x_sec_max = max(x1, x2, x_next) + 0.3
                x_sec = np.linspace(x_sec_min, x_sec_max, 100)
                y_sec = f1 + pendiente * (x_sec - x1)

                # Clipear valores extremos
                y_sec = np.clip(y_sec, -0.8, 0.8)

                self.ax_especial.plot(
                    x_sec, y_sec,
                    color=color,
                    linestyle='-',
                    linewidth=2.5,
                    alpha=0.85,
                    label=f'Secante {i+1}' if i < 4 else None
                )

            # Marcar los dos puntos de la secante con bordes blancos
            self.ax_especial.plot(x1, f1, 'o', color=color, markersize=12,
                                 markeredgecolor='white', markeredgewidth=2, zorder=5)
            self.ax_especial.plot(x2, f2, 's', color=color, markersize=10,
                                 markeredgecolor='white', markeredgewidth=2, zorder=5)

            # Marcar donde la secante cruza el eje x
            self.ax_especial.plot(x_next, 0, 'D', color=color, markersize=8,
                                 markeredgecolor='white', markeredgewidth=1.5, alpha=0.9, zorder=4)

            # Línea vertical punteada al eje x
            self.ax_especial.plot([x_next, x_next], [0, f2/3], ':', color=color, linewidth=2, alpha=0.6)

        # Marcar puntos iniciales con estilo destacado
        x0 = float(self.x0_var.get())
        x1_init = float(self.x1_var.get())
        try:
            fx0 = self.funcion_actual(x0)
            fx1 = self.funcion_actual(x1_init)
            self.ax_especial.plot(x0, fx0, 'o', color='#2c3e50', markersize=14,
                                 markeredgecolor='white', markeredgewidth=2,
                                 label=f'x₀ = {x0:.2f}', zorder=6)
            self.ax_especial.plot(x1_init, fx1, '^', color='#2c3e50', markersize=14,
                                 markeredgecolor='white', markeredgewidth=2,
                                 label=f'x₁ = {x1_init:.2f}', zorder=6)
        except:
            pass

        # Marcar raíz final con estrella grande
        if raiz:
            self.ax_especial.plot(raiz, 0, '*', color='#27ae60', markersize=25,
                                   markeredgecolor='white', markeredgewidth=2,
                                   label=f'Raíz ≈ {raiz:.6f}', zorder=10)
            # Línea vertical en la raíz
            self.ax_especial.axvline(x=raiz, color='#27ae60', linestyle=':', linewidth=2, alpha=0.7)

        # Configuración del gráfico mejorada
        self.ax_especial.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_especial.set_ylabel('P(x)', fontsize=13, fontweight='bold')
        self.ax_especial.set_title(
            f'Método de la Secante - Convergencia Visual\n'
            f'{resultado.get("iteraciones", 0)} iteraciones | '
            f'{resultado.get("evaluaciones", 0)} evaluaciones de f(x)',
            fontsize=14, fontweight='bold', pad=15
        )
        self.ax_especial.legend(loc='upper right', fontsize=10, framealpha=0.95,
                               fancybox=True, shadow=True)
        self.ax_especial.grid(True, alpha=0.4, linestyle='--')

        # Ajustar límites para mejor visualización
        self.ax_especial.set_xlim(-0.2, 1.5)
        self.ax_especial.set_ylim(-0.5, 0.5)

    def _graficar_intervalos(self, resultado: Dict[str, Any]):
        """Grafica los intervalos para bisección/falsa posición."""
        historial = resultado.get('historial', [])
        if not historial:
            return

        # Configurar fondo
        self.ax_especial.set_facecolor('#f8f9fa')

        iteraciones = [h['n'] for h in historial]
        a_vals = [h['a'] for h in historial]
        b_vals = [h['b'] for h in historial]
        c_vals = [h['c'] for h in historial]

        # Colores degradados para las barras
        n_bars = min(len(historial), 15)
        colores = plt.cm.Blues(np.linspace(0.3, 0.9, n_bars))

        # Graficar intervalos como barras con mejor estilo
        for i, (n, a, b, c) in enumerate(zip(iteraciones[:15], a_vals[:15], b_vals[:15], c_vals[:15])):
            self.ax_especial.barh(n, b - a, left=a, height=0.7, alpha=0.8,
                                 color=colores[i], edgecolor='white', linewidth=1.5)
            # Marcar el punto c con círculo rojo
            self.ax_especial.plot(c, n, 'o', color='#e74c3c', markersize=12,
                                 markeredgecolor='white', markeredgewidth=2, zorder=5)

        # Marcar raíz final con línea vertical verde
        raiz = resultado.get('raiz')
        if raiz:
            self.ax_especial.axvline(x=raiz, color='#27ae60', linestyle='--', linewidth=3,
                                      label=f'Raíz = {raiz:.6f}', zorder=10)
            # Añadir punto en la raíz
            self.ax_especial.plot(raiz, 0.5, '*', color='#27ae60', markersize=20,
                                 markeredgecolor='white', markeredgewidth=2, zorder=11)

        self.ax_especial.set_xlabel('x', fontsize=13, fontweight='bold')
        self.ax_especial.set_ylabel('Iteración', fontsize=13, fontweight='bold')

        metodo = resultado.get('metodo', 'Bisección')
        self.ax_especial.set_title(f'Reducción del Intervalo - {metodo}\n'
                                   f'{len(historial)} iteraciones hasta convergencia',
                                   fontsize=14, fontweight='bold', pad=15)
        self.ax_especial.legend(loc='upper right', fontsize=11, framealpha=0.95, fancybox=True)
        self.ax_especial.grid(True, alpha=0.4, linestyle='--', axis='x')
        self.ax_especial.invert_yaxis()

        # Añadir texto explicativo
        self.ax_especial.text(0.02, 0.02, '* Aproximación c en cada iteración',
                             transform=self.ax_especial.transAxes, fontsize=9,
                             color='#e74c3c', fontweight='bold')

    def _limpiar(self):
        """Limpia todos los campos y resultados."""
        # Limpiar texto
        self.text_resultados.config(state=tk.NORMAL)
        self.text_resultados.delete(1.0, tk.END)
        self.text_resultados.config(state=tk.DISABLED)

        # Limpiar tabla
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Limpiar gráficas
        self.ax_funcion.clear()
        self.ax_convergencia.clear()
        self.ax_especial.clear()

        self.canvas_funcion.draw()
        self.canvas_convergencia.draw()
        self.canvas_especial.draw()

        # Resetear variables
        self.resultado_actual = None
        self.funcion_actual = None
        self.derivada_actual = None

    def _comparar_metodos(self):
        """Compara bisección y falsa posición para el ejercicio 2 (Balanceo de Carga)."""
        try:
            # Cargar función del ejercicio 2
            ejercicio = FUNCIONES_EJERCICIOS["Ejercicio 2 - Falsa Posición"]
            funcion_str = ejercicio['funcion_str']
            f = obtener_funcion(funcion_str)
            self.funcion_actual = f

            a, b = ejercicio['intervalo']
            tolerancia = ejercicio['tolerancia']
            max_iter = ejercicio['max_iter']

            # Ejecutar ambos métodos
            biseccion = MetodoBiseccion(f, a, b, tolerancia, max_iter)
            resultado_bis = biseccion.resolver()

            falsa_pos = MetodoFalsaPosicion(f, a, b, tolerancia, max_iter)
            resultado_fp = falsa_pos.resolver()

            # Mostrar comparación en ventana con tabla detallada
            self._mostrar_comparacion_detallada(resultado_bis, resultado_fp, ejercicio)

            # Graficar comparación de convergencia superpuesta
            self._graficar_comparacion_superpuesta(resultado_bis, resultado_fp)

            # También actualizar la gráfica de la función
            self._graficar_funcion_comparacion(resultado_bis, resultado_fp)

        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")

    def _mostrar_comparacion_detallada(self, resultado_bis: Dict, resultado_fp: Dict, ejercicio: Dict):
        """Muestra una ventana con la comparación detallada de métodos incluyendo tabla de iteraciones."""
        ventana = tk.Toplevel(self.root)
        ventana.title("Comparación Detallada: Bisección vs Falsa Posición")
        ventana.geometry("1000x700")
        ventana.minsize(900, 600)

        # Notebook para organizar contenido
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === TAB 1: Resumen ===
        frame_resumen = ttk.Frame(notebook)
        notebook.add(frame_resumen, text="📊 Resumen")

        texto_resumen = tk.Text(frame_resumen, font=('Consolas', 11), wrap=tk.WORD)
        texto_resumen.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        comparacion = f"""
{'═'*70}
   COMPARACIÓN: BISECCIÓN vs FALSA POSICIÓN
   Problema: {ejercicio.get('descripcion', 'Balanceo de Carga')}
{'═'*70}

📐 Función: E(x) = x³ - 6x² + 11x - 6.5
📍 Intervalo: [{ejercicio['intervalo'][0]}, {ejercicio['intervalo'][1]}]
🎯 Tolerancia: {ejercicio['tolerancia']:.0e}

{'─'*70}
{'Criterio':<30} {'Bisección':<18} {'Falsa Posición':<18}
{'─'*70}
{'Raíz aproximada':<30} {resultado_bis.get('raiz', 0):<18.10f} {resultado_fp.get('raiz', 0):<18.10f}
{'Iteraciones':<30} {resultado_bis.get('iteraciones', 0):<18d} {resultado_fp.get('iteraciones', 0):<18d}
{'Error final':<30} {resultado_bis.get('error_final', 0):<18.2e} {resultado_fp.get('error_final', 0):<18.2e}
{'Tiempo (ms)':<30} {resultado_bis.get('tiempo', 0)*1000:<18.6f} {resultado_fp.get('tiempo', 0)*1000:<18.6f}
{'Evaluaciones f(x)':<30} {resultado_bis.get('evaluaciones', 0):<18d} {resultado_fp.get('evaluaciones', 0):<18d}
{'Convergió':<30} {'✓ Sí' if resultado_bis.get('convergencia') else '✗ No':<18} {'✓ Sí' if resultado_fp.get('convergencia') else '✗ No':<18}
{'─'*70}

{'═'*70}
📊 ANÁLISIS DE CONVERGENCIA
{'═'*70}

"""
        iter_bis = resultado_bis.get('iteraciones', float('inf'))
        iter_fp = resultado_fp.get('iteraciones', float('inf'))
        eval_bis = resultado_bis.get('evaluaciones', 0)
        eval_fp = resultado_fp.get('evaluaciones', 0)

        # Calcular reducción porcentual
        if iter_bis > 0:
            reduccion = ((iter_bis - iter_fp) / iter_bis) * 100
        else:
            reduccion = 0

        if iter_fp < iter_bis:
            comparacion += f"""🏆 GANADOR: FALSA POSICIÓN

   → Falsa Posición convergió en {iter_fp} iteraciones
   → Bisección convergió en {iter_bis} iteraciones
   → Reducción de {abs(reduccion):.1f}% en iteraciones

   ¿Por qué Falsa Posición es más rápido?
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Bisección SIEMPRE divide el intervalo exactamente a la mitad
   • Falsa Posición usa INTERPOLACIÓN LINEAL: c = b - f(b)·(b-a)/(f(b)-f(a))
   • Esta interpolación "predice" mejor la ubicación de la raíz
   • Converge más rápido cuando la función es aproximadamente lineal
     cerca de la raíz

"""
        elif iter_bis < iter_fp:
            comparacion += f"""🏆 GANADOR: BISECCIÓN

   → Bisección convergió en {iter_bis} iteraciones
   → Falsa Posición convergió en {iter_fp} iteraciones

   ¿Por qué Bisección ganó en este caso?
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Esto puede ocurrir cuando la función tiene pendiente muy pronunciada
   • Falsa Posición puede "estancarse" si f(a) >> f(b) o viceversa
   • En estos casos, el punto c se acerca repetidamente al mismo extremo
   • Bisección tiene convergencia garantizada y predecible

"""
        else:
            comparacion += "⚖️ Ambos métodos convergieron en el mismo número de iteraciones.\n"

        # Interpretación del resultado
        raiz = resultado_fp.get('raiz') if resultado_fp.get('convergencia') else resultado_bis.get('raiz')
        if raiz:
            comparacion += f"""
{'═'*70}
🎯 INTERPRETACIÓN DEL RESULTADO
{'═'*70}

   Número óptimo de workers: x ≈ {raiz:.6f}
   Workers recomendados (redondeado): {round(raiz)} workers

   Esto significa que para máxima eficiencia del sistema distribuido,
   se deberían tener aproximadamente {round(raiz)} workers activos.

   Nota: En la práctica, se puede elegir entre {int(raiz)} y {int(raiz)+1}
   workers dependiendo de la carga de trabajo esperada.

"""

        texto_resumen.insert(tk.END, comparacion)
        texto_resumen.config(state=tk.DISABLED)

        # === TAB 2: Tabla Comparativa de Iteraciones ===
        frame_tabla = ttk.Frame(notebook)
        notebook.add(frame_tabla, text="📋 Tabla Iteraciones")

        # Crear Treeview para tabla comparativa
        tree_frame = ttk.Frame(frame_tabla)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        columnas = ('n', 'a_bis', 'b_bis', 'c_bis', 'error_bis', 'c_fp', 'error_fp')
        tree = ttk.Treeview(tree_frame, columns=columnas, show='headings',
                           yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        # Configurar columnas
        tree.heading('n', text='n')
        tree.heading('a_bis', text='a (Bis)')
        tree.heading('b_bis', text='b (Bis)')
        tree.heading('c_bis', text='c (Bisección)')
        tree.heading('error_bis', text='Error (Bis)')
        tree.heading('c_fp', text='c (Falsa Pos)')
        tree.heading('error_fp', text='Error (FP)')

        for col in columnas:
            tree.column(col, width=110, anchor=tk.CENTER)
        tree.column('n', width=50)

        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)

        # Llenar tabla
        hist_bis = resultado_bis.get('historial', [])
        hist_fp = resultado_fp.get('historial', [])
        max_iter = max(len(hist_bis), len(hist_fp))

        for i in range(max_iter):
            valores = [i + 1]  # n

            if i < len(hist_bis):
                h_bis = hist_bis[i]
                valores.extend([
                    f"{h_bis['a']:.8f}",
                    f"{h_bis['b']:.8f}",
                    f"{h_bis['c']:.8f}",
                    f"{h_bis['error_absoluto']:.2e}"
                ])
            else:
                valores.extend(['-', '-', '-', '-'])

            if i < len(hist_fp):
                h_fp = hist_fp[i]
                valores.extend([
                    f"{h_fp['c']:.8f}",
                    f"{h_fp['error_absoluto']:.2e}"
                ])
            else:
                valores.extend(['-', '-'])

            tree.insert('', tk.END, values=valores)

        # === TAB 3: Métricas Detalladas ===
        frame_metricas = ttk.Frame(notebook)
        notebook.add(frame_metricas, text="📈 Métricas")

        texto_metricas = tk.Text(frame_metricas, font=('Consolas', 11), wrap=tk.WORD)
        texto_metricas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        metricas = f"""
{'═'*70}
MÉTRICAS DETALLADAS DE CONVERGENCIA
{'═'*70}

┌─────────────────────────────────────────────────────────────────────┐
│                        MÉTODO DE BISECCIÓN                          │
├─────────────────────────────────────────────────────────────────────┤
│ Iteraciones totales:     {resultado_bis.get('iteraciones', 0):<42d}│
│ Evaluaciones de f(x):    {resultado_bis.get('evaluaciones', 0):<42d}│
│ Tiempo de ejecución:     {resultado_bis.get('tiempo', 0)*1000:<42.6f}ms│
│ Error final:             {resultado_bis.get('error_final', 0):<42.2e}│
│ Raíz encontrada:         {resultado_bis.get('raiz', 0):<42.12f}│
│ Orden de convergencia:   Lineal (≈1)                               │
│ Tasa de reducción:       50% por iteración (intervalo)             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     MÉTODO DE FALSA POSICIÓN                        │
├─────────────────────────────────────────────────────────────────────┤
│ Iteraciones totales:     {resultado_fp.get('iteraciones', 0):<42d}│
│ Evaluaciones de f(x):    {resultado_fp.get('evaluaciones', 0):<42d}│
│ Tiempo de ejecución:     {resultado_fp.get('tiempo', 0)*1000:<42.6f}ms│
│ Error final:             {resultado_fp.get('error_final', 0):<42.2e}│
│ Raíz encontrada:         {resultado_fp.get('raiz', 0):<42.12f}│
│ Orden de convergencia:   Superlineal (1 < p < 2)                   │
│ Fórmula:                 c = b - f(b)·(b-a)/(f(b)-f(a))            │
└─────────────────────────────────────────────────────────────────────┘

{'═'*70}
FÓRMULAS UTILIZADAS
{'═'*70}

BISECCIÓN:
    c = (a + b) / 2
    • Siempre divide el intervalo exactamente a la mitad
    • Convergencia garantizada pero lenta
    • Reduce el intervalo en 50% cada iteración

FALSA POSICIÓN (Regula Falsi):
    c = b - f(b) · (b - a) / (f(b) - f(a))
    • Usa interpolación lineal entre (a, f(a)) y (b, f(b))
    • Encuentra donde la recta secante cruza el eje x
    • Generalmente más rápido, pero puede estancarse

{'═'*70}
"""
        texto_metricas.insert(tk.END, metricas)
        texto_metricas.config(state=tk.DISABLED)

    def _graficar_comparacion_superpuesta(self, resultado_bis: Dict, resultado_fp: Dict):
        """Grafica la comparación de convergencia superpuesta de ambos métodos."""
        # Limpiar figura de comparación
        self.fig_comparacion.clear()

        # Crear dos subplots
        ax1 = self.fig_comparacion.add_subplot(121)
        ax2 = self.fig_comparacion.add_subplot(122)

        # === Subplot 1: Convergencia del error (escala logarítmica) ===
        historial_bis = resultado_bis.get('historial', [])
        historial_fp = resultado_fp.get('historial', [])

        if historial_bis:
            iter_bis = [h['n'] for h in historial_bis]
            err_bis = [h['error_absoluto'] for h in historial_bis if h['error_absoluto'] > 0]
            if len(err_bis) == len(iter_bis):
                ax1.semilogy(iter_bis, err_bis, 'b-o', label='Bisección',
                            linewidth=2, markersize=6, markerfacecolor='white')

        if historial_fp:
            iter_fp = [h['n'] for h in historial_fp]
            err_fp = [h['error_absoluto'] for h in historial_fp if h['error_absoluto'] > 0]
            if len(err_fp) == len(iter_fp):
                ax1.semilogy(iter_fp, err_fp, 'r-s', label='Falsa Posición',
                            linewidth=2, markersize=6, markerfacecolor='white')

        # Línea de tolerancia
        tolerancia = 1e-7
        ax1.axhline(y=tolerancia, color='green', linestyle='--', linewidth=2,
                   label=f'Tolerancia = {tolerancia:.0e}')

        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Error Absoluto (escala log)', fontsize=11)
        ax1.set_title('Convergencia del Error\nBisección vs Falsa Posición', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, which='both')

        # === Subplot 2: Comparación de métricas (barras) ===
        metodos = ['Bisección', 'Falsa Posición']
        iteraciones = [
            resultado_bis.get('iteraciones', 0),
            resultado_fp.get('iteraciones', 0)
        ]
        evaluaciones = [
            resultado_bis.get('evaluaciones', 0),
            resultado_fp.get('evaluaciones', 0)
        ]
        tiempos = [
            resultado_bis.get('tiempo', 0) * 1000,  # ms
            resultado_fp.get('tiempo', 0) * 1000
        ]

        x_pos = np.arange(len(metodos))
        width = 0.25

        bars1 = ax2.bar(x_pos - width, iteraciones, width, label='Iteraciones', color='steelblue')
        bars2 = ax2.bar(x_pos, evaluaciones, width, label='Evaluaciones f(x)', color='coral')

        # Agregar etiquetas de valor
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.set_ylabel('Cantidad', fontsize=11)
        ax2.set_title('Comparación de Métricas', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metodos)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        # Indicar ganador
        ganador_iter = 'Falsa Posición' if iteraciones[1] < iteraciones[0] else 'Bisección'
        if iteraciones[0] == iteraciones[1]:
            ganador_iter = 'Empate'
        ax2.set_xlabel(f'* Menos iteraciones: {ganador_iter}', fontsize=10)

        self.fig_comparacion.tight_layout()
        self.canvas_comparacion.draw()

        # Cambiar a la pestaña de comparación
        self.notebook_graficas.select(3)

    def _graficar_funcion_comparacion(self, resultado_bis: Dict, resultado_fp: Dict):
        """Grafica la función con las aproximaciones de ambos métodos superpuestas."""
        self.ax_funcion.clear()

        if not self.funcion_actual:
            return

        # Rango de la gráfica
        x_min, x_max = 1.5, 4.5
        x = np.linspace(x_min, x_max, 500)

        try:
            y = [self.funcion_actual(xi) for xi in x]
            y = np.array(y)
        except:
            return

        # Graficar función
        self.ax_funcion.plot(x, y, 'b-', linewidth=2.5, label='E(x) = x³ - 6x² + 11x - 6.5')

        # Línea y=0
        self.ax_funcion.axhline(y=0, color='gray', linestyle='-', linewidth=1)

        # Obtener aproximaciones de bisección
        hist_bis = resultado_bis.get('historial', [])
        if hist_bis:
            c_bis = [h['c'] for h in hist_bis]
            y_bis = [self.funcion_actual(c) for c in c_bis]
            self.ax_funcion.scatter(c_bis, y_bis, c='blue', s=40, alpha=0.6,
                                   label=f'Bisección ({len(c_bis)} iter)', marker='o', zorder=4)

        # Obtener aproximaciones de falsa posición
        hist_fp = resultado_fp.get('historial', [])
        if hist_fp:
            c_fp = [h['c'] for h in hist_fp]
            y_fp = [self.funcion_actual(c) for c in c_fp]
            self.ax_funcion.scatter(c_fp, y_fp, c='red', s=40, alpha=0.6,
                                   label=f'Falsa Posición ({len(c_fp)} iter)', marker='s', zorder=4)

        # Marcar raíz final
        raiz = resultado_fp.get('raiz') if resultado_fp.get('convergencia') else resultado_bis.get('raiz')
        if raiz:
            self.ax_funcion.plot(raiz, 0, 'g*', markersize=20, label=f'Raíz ≈ {raiz:.6f}', zorder=5)
            self.ax_funcion.axvline(x=raiz, color='green', linestyle=':', alpha=0.5)

        self.ax_funcion.set_xlabel('x', fontsize=12)
        self.ax_funcion.set_ylabel('E(x)', fontsize=12)
        self.ax_funcion.set_title('Función E(x) con Aproximaciones de Ambos Métodos\n(Balanceo de Carga)', fontsize=13)
        self.ax_funcion.legend(loc='best')
        self.ax_funcion.grid(True, alpha=0.3)

        self.fig_funcion.tight_layout()
        self.canvas_funcion.draw()

    def _mostrar_comparacion(self, resultado_bis: Dict, resultado_fp: Dict):
        """Muestra una ventana con la comparación de métodos."""
        ventana = tk.Toplevel(self.root)
        ventana.title("Comparación de Métodos")
        ventana.geometry("600x400")

        texto = tk.Text(ventana, font=('Consolas', 11), wrap=tk.WORD)
        texto.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        comparacion = f"""
{'='*60}
COMPARACIÓN: BISECCIÓN vs FALSA POSICIÓN
{'='*60}

{'Criterio':<25} {'Bisección':<15} {'Falsa Posición':<15}
{'-'*60}
{'Raíz aproximada':<25} {resultado_bis.get('raiz', 0):<15.10f} {resultado_fp.get('raiz', 0):<15.10f}
{'Iteraciones':<25} {resultado_bis.get('iteraciones', 0):<15d} {resultado_fp.get('iteraciones', 0):<15d}
{'Error final':<25} {resultado_bis.get('error_final', 0):<15.2e} {resultado_fp.get('error_final', 0):<15.2e}
{'Tiempo (ms)':<25} {resultado_bis.get('tiempo', 0)*1000:<15.4f} {resultado_fp.get('tiempo', 0)*1000:<15.4f}
{'Evaluaciones f(x)':<25} {resultado_bis.get('evaluaciones', 0):<15d} {resultado_fp.get('evaluaciones', 0):<15d}
{'Convergió':<25} {'Sí' if resultado_bis.get('convergencia') else 'No':<15} {'Sí' if resultado_fp.get('convergencia') else 'No':<15}

{'='*60}
ANÁLISIS
{'='*60}

"""

        # Análisis
        iter_bis = resultado_bis.get('iteraciones', float('inf'))
        iter_fp = resultado_fp.get('iteraciones', float('inf'))

        if iter_fp < iter_bis:
            comparacion += f"""✅ Falsa Posición convergió más rápido ({iter_fp} vs {iter_bis} iteraciones).

Esto se debe a que Falsa Posición usa interpolación lineal para estimar
la raíz, lo que generalmente da mejores aproximaciones que simplemente
dividir el intervalo a la mitad (Bisección).
"""
        elif iter_bis < iter_fp:
            comparacion += f"""✅ Bisección convergió más rápido ({iter_bis} vs {iter_fp} iteraciones).

Esto puede ocurrir cuando la función tiene una pendiente muy pronunciada
cerca de uno de los extremos, lo que puede causar que Falsa Posición
se "estanque" en un lado del intervalo.
"""
        else:
            comparacion += "⚖️ Ambos métodos convergieron en el mismo número de iteraciones."

        texto.insert(tk.END, comparacion)
        texto.config(state=tk.DISABLED)

    def _graficar_comparacion(self, resultado_bis: Dict, resultado_fp: Dict):
        """Grafica la comparación de convergencia."""
        self.ax_especial.clear()

        # Obtener errores
        historial_bis = resultado_bis.get('historial', [])
        historial_fp = resultado_fp.get('historial', [])

        if historial_bis:
            iter_bis = [h['n'] for h in historial_bis]
            err_bis = [h['error_absoluto'] for h in historial_bis]
            self.ax_especial.semilogy(iter_bis, err_bis, 'b-o', label='Bisección', linewidth=2)

        if historial_fp:
            iter_fp = [h['n'] for h in historial_fp]
            err_fp = [h['error_absoluto'] for h in historial_fp]
            self.ax_especial.semilogy(iter_fp, err_fp, 'r-s', label='Falsa Posición', linewidth=2)

        self.ax_especial.set_xlabel('Iteración', fontsize=12)
        self.ax_especial.set_ylabel('Error Absoluto (log)', fontsize=12)
        self.ax_especial.set_title('Comparación de Convergencia', fontsize=14)
        self.ax_especial.legend(loc='best')
        self.ax_especial.grid(True, alpha=0.3, which='both')

        self.fig_especial.tight_layout()
        self.canvas_especial.draw()

        # Cambiar a la pestaña especial
        self.notebook_graficas.select(2)

    def _comparar_newton_secante(self):
        """
        Compara los métodos Newton-Raphson y Secante.

        Utiliza la función del Ejercicio 5 para comparar ambos métodos
        en términos de iteraciones, evaluaciones de función y precisión.
        Implementa las tareas 19-22 del Ejercicio 5.
        """
        try:
            # Usar la función del ejercicio 5
            ejercicio = FUNCIONES_EJERCICIOS["Ejercicio 5 - Secante"]
            funcion_str = ejercicio['funcion_str']
            f = obtener_funcion(funcion_str)
            df = obtener_derivada(funcion_str)

            x0 = ejercicio['x0']
            x1 = ejercicio['x1']
            tolerancia = ejercicio['tolerancia']
            max_iter = ejercicio['max_iter']

            # Ejecutar Newton-Raphson
            newton = MetodoNewton(f, df, x0, tolerancia, max_iter)
            resultado_newton = newton.resolver()

            # Ejecutar Secante
            secante = MetodoSecante(f, x0, x1, tolerancia, max_iter)
            resultado_secante = secante.resolver()

            # Guardar función actual para gráficas
            self.funcion_actual = f

            # Mostrar comparación en ventana con tabs
            self._mostrar_comparacion_newton_secante(resultado_newton, resultado_secante, ejercicio)

            # Graficar comparación
            self._graficar_comparacion_newton_secante(resultado_newton, resultado_secante)

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Error en comparación Newton vs Secante:\n{str(e)}\n\n{traceback.format_exc()}")

    def _mostrar_comparacion_newton_secante(self, resultado_newton: Dict, resultado_secante: Dict, ejercicio: Dict):
        """Muestra ventana con comparación detallada Newton vs Secante con tabs."""
        ventana = tk.Toplevel(self.root)
        ventana.title("Comparación Newton-Raphson vs Secante - Ejercicio 5")
        ventana.geometry("1000x700")
        ventana.minsize(900, 600)

        # Notebook para organizar contenido
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === TAB 1: Resumen Comparativo ===
        frame_resumen = ttk.Frame(notebook)
        notebook.add(frame_resumen, text="📊 Resumen")

        texto_resumen = tk.Text(frame_resumen, font=('Consolas', 10), wrap=tk.WORD)
        scrollbar_resumen = ttk.Scrollbar(frame_resumen, orient=tk.VERTICAL, command=texto_resumen.yview)
        texto_resumen.configure(yscrollcommand=scrollbar_resumen.set)
        scrollbar_resumen.pack(side=tk.RIGHT, fill=tk.Y)
        texto_resumen.pack(fill=tk.BOTH, expand=True)

        # Obtener valores con manejo de None
        raiz_n = resultado_newton.get('raiz', 0) or 0
        raiz_s = resultado_secante.get('raiz', 0) or 0

        resumen = f"""
{'='*75}
EJERCICIO 5: PREDICCIÓN DE ESCALABILIDAD CON MÉTODO DE LA SECANTE
{'='*75}

📐 FUNCIÓN: P(x) = x·e^(-x/2) - 0.3
   Derivada: P'(x) = e^(-x/2)·(1 - x/2)

📋 CONTEXTO:
   Una plataforma cloud necesita predecir el punto donde el costo de
   infraestructura iguala los ingresos. x = miles de usuarios activos.

{'='*75}
TABLA COMPARATIVA
{'='*75}

{'Criterio':<30} {'Newton-Raphson':<20} {'Secante':<20}
{'-'*75}
{'Raíz aproximada':<30} {raiz_n:<20.12f} {raiz_s:<20.12f}
{'Iteraciones':<30} {resultado_newton.get('iteraciones', 0):<20d} {resultado_secante.get('iteraciones', 0):<20d}
{'Error final':<30} {resultado_newton.get('error_final', 0):<20.2e} {resultado_secante.get('error_final', 0):<20.2e}
{'Tiempo (ms)':<30} {resultado_newton.get('tiempo', 0)*1000:<20.6f} {resultado_secante.get('tiempo', 0)*1000:<20.6f}
{'Evaluaciones f(x)':<30} {resultado_newton.get('evaluaciones_f', resultado_newton.get('evaluaciones', 0)):<20d} {resultado_secante.get('evaluaciones', 0):<20d}
{"Evaluaciones f'(x)":<30} {resultado_newton.get('evaluaciones_df', 0):<20d} {'N/A (no requiere)':<20}
{'Total evaluaciones':<30} {resultado_newton.get('evaluaciones', 0):<20d} {resultado_secante.get('evaluaciones', 0):<20d}
{'Convergió':<30} {'✓ Sí' if resultado_newton.get('convergencia') else '✗ No':<20} {'✓ Sí' if resultado_secante.get('convergencia') else '✗ No':<20}

"""

        # Análisis detallado
        iter_n = resultado_newton.get('iteraciones', float('inf'))
        iter_s = resultado_secante.get('iteraciones', float('inf'))
        eval_n = resultado_newton.get('evaluaciones', 0)
        eval_s = resultado_secante.get('evaluaciones', 0)

        resumen += f"""
{'='*75}
📊 ANÁLISIS COMPARATIVO
{'='*75}

📈 VELOCIDAD DE CONVERGENCIA:
"""
        if iter_n < iter_s:
            resumen += f"   → Newton-Raphson convergió más rápido ({iter_n} vs {iter_s} iteraciones)\n"
            resumen += "   → Esperado: Newton tiene convergencia CUADRÁTICA (orden 2)\n"
            resumen += "   → Secante tiene convergencia de orden φ ≈ 1.618 (número áureo)\n"
        elif iter_s < iter_n:
            resumen += f"   → Secante convergió más rápido ({iter_s} vs {iter_n} iteraciones)\n"
            resumen += "   → Esto es inusual, podría deberse a la elección de x₀\n"
        else:
            resumen += f"   → Ambos convergieron en {iter_n} iteraciones\n"

        resumen += f"""
📉 COSTO COMPUTACIONAL:
   • Newton-Raphson: {resultado_newton.get('evaluaciones_f', 0)} eval. f(x) + {resultado_newton.get('evaluaciones_df', 0)} eval. f'(x) = {eval_n} total
   • Secante: {eval_s} evaluaciones de f(x) (sin derivadas)
"""
        if eval_s < eval_n:
            resumen += f"   → Secante requirió MENOS evaluaciones totales ({eval_s} vs {eval_n})\n"
        else:
            resumen += f"   → Newton requirió menos evaluaciones ({eval_n} vs {eval_s})\n"

        resumen += f"""
{'='*75}
🎯 ANÁLISIS: ¿VALE LA PENA CALCULAR DERIVADAS ANALÍTICAS?
{'='*75}

Para P(x) = x·e^(-x/2) - 0.3:
   • Derivada: P'(x) = e^(-x/2)·(1 - x/2)

✅ ARGUMENTOS A FAVOR DE NEWTON (usar derivadas):
   • Convergencia cuadrática (más rápida por iteración)
   • La derivada es algebraicamente simple
   • Menos iteraciones necesarias

✅ ARGUMENTOS A FAVOR DE SECANTE (sin derivadas):
   • No requiere cálculo simbólico de derivadas
   • Más fácil de implementar para funciones complejas
   • Ideal para funciones definidas por datos/tablas
   • Cada iteración es más barata (solo evalúa f, no f')

💡 CONCLUSIÓN PARA ESTE PROBLEMA:
   La derivada P'(x) = e^(-x/2)·(1 - x/2) es relativamente simple.
"""
        if eval_n <= eval_s and iter_n <= iter_s:
            resumen += "   → Newton-Raphson es preferible para este caso específico.\n"
        elif eval_s < eval_n:
            resumen += "   → Secante resultó más eficiente en evaluaciones totales.\n"
        else:
            resumen += "   → Ambos métodos son comparables para este problema.\n"

        resumen += f"""
   Sin embargo, si la función fuera más compleja (datos empíricos,
   integrales, etc.), la Secante sería la opción más práctica.

{'='*75}
📍 INTERPRETACIÓN DEL RESULTADO
{'='*75}

   Raíz encontrada: x ≈ {raiz_s:.10f}

   Esto significa que el punto de equilibrio financiero ocurre cuando
   hay aproximadamente {raiz_s*1000:.0f} usuarios activos.

   • Menos de {raiz_s*1000:.0f} usuarios → Pérdidas (costos > ingresos)
   • Más de {raiz_s*1000:.0f} usuarios → Ganancias (ingresos > costos)
"""

        texto_resumen.insert(tk.END, resumen)
        texto_resumen.config(state=tk.DISABLED)

        # === TAB 2: Tabla de Iteraciones Secante ===
        frame_iter_secante = ttk.Frame(notebook)
        notebook.add(frame_iter_secante, text="📋 Iteraciones Secante")

        # Crear Treeview para Secante
        cols_secante = ('n', 'x_{n-1}', 'x_n', 'f(x_{n-1})', 'f(x_n)', 'x_{n+1}', 'Error')
        tree_secante = ttk.Treeview(frame_iter_secante, columns=cols_secante, show='headings', height=20)

        for col in cols_secante:
            tree_secante.heading(col, text=col)
            tree_secante.column(col, width=120, anchor='center')

        scrollbar_s = ttk.Scrollbar(frame_iter_secante, orient=tk.VERTICAL, command=tree_secante.yview)
        tree_secante.configure(yscrollcommand=scrollbar_s.set)

        tree_secante.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_s.pack(side=tk.RIGHT, fill=tk.Y)

        # Llenar tabla Secante
        for h in resultado_secante.get('historial', []):
            tree_secante.insert('', tk.END, values=(
                h.get('n', ''),
                f"{h.get('x_{n-1}', 0):.10f}",
                f"{h.get('x_n', 0):.10f}",
                f"{h.get('f(x_{n-1})', 0):.6e}",
                f"{h.get('f(x_n)', 0):.6e}",
                f"{h.get('x_{n+1}', 0):.10f}",
                f"{h.get('error_absoluto', 0):.6e}"
            ))

        # === TAB 3: Tabla de Iteraciones Newton ===
        frame_iter_newton = ttk.Frame(notebook)
        notebook.add(frame_iter_newton, text="📋 Iteraciones Newton")

        # Crear Treeview para Newton
        cols_newton = ('n', 'x_n', 'f(x_n)', "f'(x_n)", 'x_{n+1}', 'Error Abs', 'Error Rel')
        tree_newton = ttk.Treeview(frame_iter_newton, columns=cols_newton, show='headings', height=20)

        for col in cols_newton:
            tree_newton.heading(col, text=col)
            tree_newton.column(col, width=110, anchor='center')

        scrollbar_n = ttk.Scrollbar(frame_iter_newton, orient=tk.VERTICAL, command=tree_newton.yview)
        tree_newton.configure(yscrollcommand=scrollbar_n.set)

        tree_newton.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_n.pack(side=tk.RIGHT, fill=tk.Y)

        # Llenar tabla Newton
        for h in resultado_newton.get('historial', []):
            tree_newton.insert('', tk.END, values=(
                h.get('n', ''),
                f"{h.get('x_n', 0):.10f}",
                f"{h.get('f(x_n)', 0):.6e}",
                f"{h.get('f_prime_xn', h.get('fpx', 0)):.6e}",
                f"{h.get('x_n+1', 0):.10f}",
                f"{h.get('error_absoluto', 0):.6e}",
                f"{h.get('error_relativo', 0):.6e}"
            ))

        # === TAB 4: Gráficas ===
        frame_graficas = ttk.Frame(notebook)
        notebook.add(frame_graficas, text="📈 Gráficas")

        # Crear figura con subplots
        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame_graficas)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Subplot 1: Función con secantes
        ax1 = fig.add_subplot(221)
        x_vals = np.linspace(0.1, 3.0, 200)
        try:
            f = obtener_funcion(ejercicio['funcion_str'])
            y_vals = [f(x) for x in x_vals]
            ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='P(x) = x·e^(-x/2) - 0.3')
            ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

            # Dibujar secantes
            colores = plt.cm.Reds(np.linspace(0.3, 0.9, len(resultado_secante.get('secantes', []))))
            for i, sec in enumerate(resultado_secante.get('secantes', [])[:5]):  # Primeras 5
                x1, x2 = sec['x_prev'], sec['x_curr']
                y1, y2 = sec['f_prev'], sec['f_curr']
                # Extender la línea secante
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    x_line = np.linspace(min(x1, x2) - 0.3, max(x1, x2) + 0.3, 50)
                    y_line = y1 + m * (x_line - x1)
                    ax1.plot(x_line, y_line, '--', color=colores[i], alpha=0.7, linewidth=1.5)
                ax1.plot([x1, x2], [y1, y2], 'o', color=colores[i], markersize=6)

            # Marcar raíz
            if resultado_secante.get('raiz'):
                ax1.axvline(x=resultado_secante['raiz'], color='g', linestyle='--', alpha=0.7)
                ax1.plot(resultado_secante['raiz'], 0, 'g*', markersize=15, label=f"Raíz ≈ {resultado_secante['raiz']:.6f}")

            ax1.set_xlabel('x')
            ax1.set_ylabel('P(x)')
            ax1.set_title('Función con Secantes')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')

        # Subplot 2: Convergencia del error
        ax2 = fig.add_subplot(222)
        historial_n = resultado_newton.get('historial', [])
        historial_s = resultado_secante.get('historial', [])

        if historial_n:
            iter_n = [h['n'] for h in historial_n]
            err_n = [h['error_absoluto'] for h in historial_n if h['error_absoluto'] > 0]
            if err_n:
                ax2.semilogy(iter_n[:len(err_n)], err_n, 'b-o', label='Newton-Raphson', linewidth=2, markersize=6)

        if historial_s:
            iter_s = [h['n'] for h in historial_s]
            err_s = [h['error_absoluto'] for h in historial_s if h['error_absoluto'] > 0]
            if err_s:
                ax2.semilogy(iter_s[:len(err_s)], err_s, 'r-s', label='Secante', linewidth=2, markersize=6)

        ax2.axhline(y=ejercicio['tolerancia'], color='green', linestyle='--', linewidth=2, label=f"Tol = {ejercicio['tolerancia']:.0e}")
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Error Absoluto (log)')
        ax2.set_title('Convergencia del Error')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, which='both')

        # Subplot 3: Barras comparativas
        ax3 = fig.add_subplot(223)
        metodos = ['Newton', 'Secante']
        iteraciones = [resultado_newton.get('iteraciones', 0), resultado_secante.get('iteraciones', 0)]
        evaluaciones = [resultado_newton.get('evaluaciones', 0), resultado_secante.get('evaluaciones', 0)]

        x_pos = np.arange(len(metodos))
        width = 0.35

        bars1 = ax3.bar(x_pos - width/2, iteraciones, width, label='Iteraciones', color='steelblue')
        bars2 = ax3.bar(x_pos + width/2, evaluaciones, width, label='Evaluaciones', color='coral')

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

        ax3.set_ylabel('Cantidad')
        ax3.set_title('Comparación de Métricas')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metodos)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3, axis='y')

        # Subplot 4: Orden de convergencia
        ax4 = fig.add_subplot(224)

        # Verificar orden de convergencia para Newton (debería ser ~2)
        if len(historial_n) >= 3:
            err_n = [h['error_absoluto'] for h in historial_n if h['error_absoluto'] > 1e-15]
            if len(err_n) >= 3:
                ratios_n = [err_n[i+1]/(err_n[i]**2) if err_n[i] > 1e-15 else 0 for i in range(len(err_n)-1)]
                ax4.plot(range(1, len(ratios_n)+1), ratios_n, 'b-o', label='Newton: e_{n+1}/e_n²', linewidth=2)

        # Verificar orden para Secante (debería ser ~1.618)
        if len(historial_s) >= 3:
            err_s = [h['error_absoluto'] for h in historial_s if h['error_absoluto'] > 1e-15]
            if len(err_s) >= 3:
                phi = 1.618
                ratios_s = [err_s[i+1]/(err_s[i]**phi) if err_s[i] > 1e-15 else 0 for i in range(len(err_s)-1)]
                ax4.plot(range(1, len(ratios_s)+1), ratios_s, 'r-s', label=f'Secante: e_{{n+1}}/e_n^φ', linewidth=2)

        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('Ratio de Convergencia')
        ax4.set_title('Verificación Orden de Convergencia')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

    def _graficar_comparacion_newton_secante(self, resultado_newton: Dict, resultado_secante: Dict):
        """Grafica la comparación de convergencia Newton vs Secante."""
        self.ax_comparacion.clear()

        # Crear subplots
        self.fig_comparacion.clear()
        ax1 = self.fig_comparacion.add_subplot(121)
        ax2 = self.fig_comparacion.add_subplot(122)

        # Subplot 1: Convergencia del error
        historial_n = resultado_newton.get('historial', [])
        historial_s = resultado_secante.get('historial', [])

        if historial_n:
            iter_n = [h['n'] for h in historial_n]
            err_n = [h['error_absoluto'] for h in historial_n if h['error_absoluto'] > 0]
            if len(err_n) == len(iter_n):
                ax1.semilogy(iter_n, err_n, 'b-o', label='Newton-Raphson', linewidth=2, markersize=8)

        if historial_s:
            iter_s = [h['n'] for h in historial_s]
            err_s = [h['error_absoluto'] for h in historial_s if h['error_absoluto'] > 0]
            if len(err_s) == len(iter_s):
                ax1.semilogy(iter_s, err_s, 'r-s', label='Secante', linewidth=2, markersize=8)

        # Línea de tolerancia
        tolerancia = 1e-9  # Tolerancia del ejercicio 5
        ax1.axhline(y=tolerancia, color='green', linestyle='--', linewidth=2, label=f'Tolerancia = {tolerancia:.0e}')

        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Error Absoluto (escala log)', fontsize=11)
        ax1.set_title('Convergencia del Error', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, which='both')

        # Subplot 2: Comparación de métricas
        metodos = ['Newton-Raphson', 'Secante']
        iteraciones = [
            resultado_newton.get('iteraciones', 0),
            resultado_secante.get('iteraciones', 0)
        ]
        evaluaciones = [
            resultado_newton.get('evaluaciones', 0),
            resultado_secante.get('evaluaciones', 0)
        ]

        x_pos = np.arange(len(metodos))
        width = 0.35

        bars1 = ax2.bar(x_pos - width/2, iteraciones, width, label='Iteraciones', color='steelblue')
        bars2 = ax2.bar(x_pos + width/2, evaluaciones, width, label='Evaluaciones', color='coral')

        # Agregar etiquetas de valor
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Cantidad', fontsize=11)
        ax2.set_title('Comparación de Métricas', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metodos)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        self.fig_comparacion.tight_layout()
        self.canvas_comparacion.draw()

        # Cambiar a la pestaña de comparación
        self.notebook_graficas.select(3)

    def _graficar_comparacion_automatica_secante(self, resultado_secante: Dict, funcion_str: str, tolerancia: float, max_iter: int):
        """
        Genera automáticamente la gráfica comparativa Newton vs Secante
        cuando se calcula con el método de la Secante.
        """
        try:
            # Obtener derivada para Newton
            df = obtener_derivada(funcion_str)
            f = self.funcion_actual

            # Usar el mismo x0 que se usó para la secante
            x0 = float(self.x0_var.get())

            # Ejecutar Newton-Raphson para comparar
            newton = MetodoNewton(f, df, x0, tolerancia, max_iter)
            resultado_newton = newton.resolver()

            # Limpiar la figura de comparación
            self.fig_comparacion.clear()

            # Configurar fondo
            self.fig_comparacion.patch.set_facecolor('#f8f9fa')

            # Crear layout 2x2 con más espacio
            ax1 = self.fig_comparacion.add_subplot(221)  # Función con secantes
            ax2 = self.fig_comparacion.add_subplot(222)  # Convergencia del error
            ax3 = self.fig_comparacion.add_subplot(223)  # Barras comparativas
            ax4 = self.fig_comparacion.add_subplot(224)  # Trayectoria de aproximaciones

            # Colores mejorados
            color_newton = '#2E86AB'  # Azul más vibrante
            color_secante = '#E94F37'  # Rojo coral
            color_raiz = '#28A745'    # Verde éxito
            color_tol = '#FFC107'     # Amarillo advertencia

            # === Subplot 1: Función con líneas secantes ===
            ax1.set_facecolor('#ffffff')
            x_vals = np.linspace(-0.2, 2.5, 300)
            try:
                y_vals = [f(x) for x in x_vals]
                ax1.plot(x_vals, y_vals, color='#1a1a2e', linewidth=2.5, label='P(x) = x·e^(-x/2) - 0.3')
                ax1.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
                ax1.fill_between(x_vals, y_vals, 0, alpha=0.1, color='blue')

                # Dibujar secantes con colores más visibles
                secantes = resultado_secante.get('secantes', [])
                colores_sec = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                for i, sec in enumerate(secantes[:5]):
                    x1, x2 = sec['x_prev'], sec['x_curr']
                    y1, y2 = sec['f_prev'], sec['f_curr']
                    if abs(x2 - x1) > 1e-10:
                        m = (y2 - y1) / (x2 - x1)
                        x_line = np.linspace(min(x1, x2) - 0.3, max(x1, x2) + 0.3, 50)
                        y_line = y1 + m * (x_line - x1)
                        y_line = np.clip(y_line, -0.6, 0.6)
                        ax1.plot(x_line, y_line, '-', color=colores_sec[i], alpha=0.9,
                                linewidth=2, label=f'Secante {i+1}' if i < 3 else None)
                    ax1.plot(x1, y1, 'o', color=colores_sec[i], markersize=10,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=5)
                    ax1.plot(x2, y2, 's', color=colores_sec[i], markersize=8,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=5)

                # Marcar raíz con estrella grande
                raiz = resultado_secante.get('raiz')
                if raiz:
                    ax1.plot(raiz, 0, '*', color=color_raiz, markersize=20,
                            markeredgecolor='white', markeredgewidth=1.5,
                            label=f'Raíz ≈ {raiz:.4f}', zorder=10)
                    ax1.axvline(x=raiz, color=color_raiz, linestyle=':', alpha=0.7, linewidth=2)

                ax1.set_xlabel('x', fontsize=11, fontweight='bold')
                ax1.set_ylabel('P(x)', fontsize=11, fontweight='bold')
                ax1.set_title('Método de la Secante', fontsize=12, fontweight='bold', pad=10)
                ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
                ax1.grid(True, alpha=0.4, linestyle='--')
                ax1.set_xlim(-0.1, 1.5)
                ax1.set_ylim(-0.4, 0.4)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax1.transAxes)

            # === Subplot 2: Convergencia del error ===
            ax2.set_facecolor('#ffffff')
            historial_n = resultado_newton.get('historial', [])
            historial_s = resultado_secante.get('historial', [])

            if historial_n:
                iter_n = [h['n'] for h in historial_n]
                err_n = [h['error_absoluto'] for h in historial_n if h['error_absoluto'] > 0]
                if len(err_n) == len(iter_n):
                    ax2.semilogy(iter_n, err_n, '-o', color=color_newton,
                                label='Newton-Raphson', linewidth=2.5, markersize=10,
                                markerfacecolor='white', markeredgewidth=2)

            if historial_s:
                iter_s = [h['n'] for h in historial_s]
                err_s = [h['error_absoluto'] for h in historial_s if h['error_absoluto'] > 0]
                if len(err_s) == len(iter_s):
                    ax2.semilogy(iter_s, err_s, '-s', color=color_secante,
                                label='Secante', linewidth=2.5, markersize=10,
                                markerfacecolor='white', markeredgewidth=2)

            ax2.axhline(y=tolerancia, color=color_tol, linestyle='--', linewidth=3,
                       label=f'Tolerancia = {tolerancia:.0e}')

            # Sombrear zona de convergencia
            ax2.axhspan(0, tolerancia, alpha=0.2, color=color_raiz, label='Zona convergida')

            ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Error Absoluto (log)', fontsize=11, fontweight='bold')
            ax2.set_title('Convergencia del Error', fontsize=12, fontweight='bold', pad=10)
            ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax2.grid(True, alpha=0.4, which='both', linestyle='--')

            # === Subplot 3: Barras comparativas ===
            ax3.set_facecolor('#ffffff')
            metodos = ['Newton-Raphson', 'Secante']
            iteraciones = [resultado_newton.get('iteraciones', 0), resultado_secante.get('iteraciones', 0)]
            evaluaciones = [resultado_newton.get('evaluaciones', 0), resultado_secante.get('evaluaciones', 0)]

            x_pos = np.arange(len(metodos))
            width = 0.35

            bars1 = ax3.bar(x_pos - width/2, iteraciones, width, label='Iteraciones',
                           color=color_newton, edgecolor='white', linewidth=2)
            bars2 = ax3.bar(x_pos + width/2, evaluaciones, width, label='Evaluaciones f(x)',
                           color=color_secante, edgecolor='white', linewidth=2)

            # Etiquetas más visibles
            for bar in bars1:
                height = bar.get_height()
                ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                            fontsize=12, fontweight='bold', color=color_newton)
            for bar in bars2:
                height = bar.get_height()
                ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                            fontsize=12, fontweight='bold', color=color_secante)

            ax3.set_ylabel('Cantidad', fontsize=11, fontweight='bold')
            ax3.set_title('Comparación de Métricas', fontsize=12, fontweight='bold', pad=10)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(metodos, fontsize=10, fontweight='bold')
            ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
            ax3.set_ylim(0, max(max(iteraciones), max(evaluaciones)) * 1.25)

            # === Subplot 4: Trayectoria de aproximaciones ===
            ax4.set_facecolor('#ffffff')
            if historial_n:
                aprox_n = [x0] + [h.get('x_n+1', h.get('x_n', 0)) for h in historial_n]
                ax4.plot(range(len(aprox_n)), aprox_n, '-o', color=color_newton,
                        label='Newton-Raphson', linewidth=2.5, markersize=10,
                        markerfacecolor='white', markeredgewidth=2)

            if historial_s:
                aprox_s = [float(self.x0_var.get()), float(self.x1_var.get())] + [h.get('x_{n+1}', h.get('x_n', 0)) for h in historial_s]
                ax4.plot(range(len(aprox_s)), aprox_s, '-s', color=color_secante,
                        label='Secante', linewidth=2.5, markersize=10,
                        markerfacecolor='white', markeredgewidth=2)

            raiz = resultado_secante.get('raiz')
            if raiz:
                ax4.axhline(y=raiz, color=color_raiz, linestyle='--', linewidth=3,
                           label=f'Raíz = {raiz:.6f}')
                # Sombrear banda alrededor de la raíz
                ax4.axhspan(raiz - 0.01, raiz + 0.01, alpha=0.2, color=color_raiz)

            ax4.set_xlabel('Iteración', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Aproximación xₙ', fontsize=11, fontweight='bold')
            ax4.set_title('Trayectoria hacia la Raíz', fontsize=12, fontweight='bold', pad=10)
            ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax4.grid(True, alpha=0.4, linestyle='--')
            ax4.grid(True, alpha=0.3)

            self.fig_comparacion.tight_layout()
            self.canvas_comparacion.draw()

        except Exception as e:
            # Si hay error, simplemente no mostrar la comparación
            pass

    def _comparar_valores_iniciales(self):
        """
        Compara la convergencia con diferentes valores iniciales.

        Funciona para:
        - Punto Fijo (Ejercicio 3)
        - Newton-Raphson (Ejercicio 4)
        """
        metodo_actual = self.metodo_var.get()

        if metodo_actual == "Newton-Raphson":
            self._comparar_x0_newton()
        elif metodo_actual == "Punto Fijo":
            self._comparar_x0_punto_fijo()
        else:
            messagebox.showinfo(
                "Información",
                "Esta función está disponible para los métodos:\n"
                "• Punto Fijo (Ejercicio 3)\n"
                "• Newton-Raphson (Ejercicio 4)\n\n"
                "Seleccione uno de estos métodos primero."
            )

    def _comparar_x0_punto_fijo(self):
        """Compara diferentes valores iniciales para Punto Fijo."""
        try:
            # Cargar función del ejercicio 3
            ejercicio = FUNCIONES_EJERCICIOS["Ejercicio 3 - Punto Fijo"]
            funcion_str = ejercicio['funcion_str']
            g = obtener_funcion(funcion_str)
            self.funcion_actual = g

            # Obtener derivada de g para verificar convergencia
            derivada_g_str = ejercicio.get('derivada_g_str', '-0.5*sin(x)')
            dg = obtener_derivada(ejercicio['funcion_str'])

            tolerancia = ejercicio['tolerancia']
            max_iter = ejercicio['max_iter']
            x0_valores = ejercicio['x0_alternativos']

            # Ejecutar punto fijo con cada x0
            resultados = {}
            for x0 in x0_valores:
                metodo = MetodoPuntoFijo(g, x0, tolerancia, max_iter, dg)
                resultados[x0] = metodo.resolver()

            # Mostrar comparación detallada
            self._mostrar_comparacion_x0(resultados, ejercicio)

            # Graficar comparación
            self._graficar_comparacion_x0(resultados, g, ejercicio)

        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación de valores iniciales: {str(e)}")

    def _comparar_x0_newton(self):
        """Compara diferentes valores iniciales para Newton-Raphson."""
        try:
            # Cargar función del ejercicio 4
            ejercicio = FUNCIONES_EJERCICIOS["Ejercicio 4 - Newton-Raphson"]
            funcion_str = ejercicio['funcion_str']
            derivada_str = ejercicio['derivada_str']
            f = obtener_funcion(funcion_str)
            df = obtener_derivada(funcion_str)
            self.funcion_actual = f

            tolerancia = ejercicio['tolerancia']
            max_iter = ejercicio['max_iter']
            x0_valores = ejercicio['x0_alternativos']

            # Ejecutar Newton con cada x0
            resultados = {}
            for x0 in x0_valores:
                metodo = MetodoNewton(f, df, x0, tolerancia, max_iter)
                resultados[x0] = metodo.resolver()

            # Mostrar comparación detallada
            self._mostrar_comparacion_x0_newton(resultados, ejercicio)

        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación de valores iniciales: {str(e)}")

    def _mostrar_comparacion_x0_newton(self, resultados: Dict, ejercicio: Dict):
        """Muestra ventana con comparación de diferentes valores iniciales para Newton-Raphson."""
        ventana = tk.Toplevel(self.root)
        ventana.title("Comparación de Valores Iniciales - Newton-Raphson")
        ventana.geometry("950x750")
        ventana.minsize(900, 650)

        # Notebook para organizar contenido
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === TAB 1: Resumen Comparativo ===
        frame_resumen = ttk.Frame(notebook)
        notebook.add(frame_resumen, text="📊 Resumen")

        texto = tk.Text(frame_resumen, font=('Consolas', 11), wrap=tk.WORD)
        scroll = ttk.Scrollbar(frame_resumen, command=texto.yview)
        texto.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        texto.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        comparacion = f"""
{'═'*80}
   COMPARACIÓN DE VALORES INICIALES - MÉTODO DE NEWTON-RAPHSON
   {ejercicio.get('descripcion', '')}
{'═'*80}

📐 Función: T(n) = n³ - 8n² + 20n - 16
📍 Derivada: T'(n) = 3n² - 16n + 20
🎯 Tolerancia: {ejercicio['tolerancia']:.0e}

{'─'*80}
⚠️  NOTA IMPORTANTE SOBRE ESTA FUNCIÓN:
{'─'*80}
   T(n) = n³ - 8n² + 20n - 16 = (n-2)²(n-4)
   
   • Raíz DOBLE en n = 2 (donde T'(2) = 0)
   • Raíz SIMPLE en n = 4
   
   Newton-Raphson converge LENTAMENTE a raíces múltiples porque
   la derivada se anula en el punto raíz.
{'─'*80}

{'─'*80}
{'x₀':<8} {'Iteraciones':<14} {'Raíz encontrada':<22} {'Error final':<15} {'Convergió':<10}
{'─'*80}
"""

        # Agregar datos de cada x0
        for x0, resultado in resultados.items():
            raiz = resultado.get('raiz', 0)
            iter_count = resultado.get('iteraciones', 0)
            error = resultado.get('error_final', 0)
            conv = "✓ Sí" if resultado.get('convergencia') else "✗ No"
            comparacion += f"{x0:<8.1f} {iter_count:<14d} {raiz:<22.12f} {error:<15.2e} {conv:<10}\n"

        comparacion += f"{'─'*80}\n"

        # Análisis de convergencia a diferentes raíces
        raices_2 = [(x0, r) for x0, r in resultados.items() if abs(r.get('raiz', 0) - 2) < 0.1]
        raices_4 = [(x0, r) for x0, r in resultados.items() if abs(r.get('raiz', 0) - 4) < 0.1]

        comparacion += f"""
{'═'*80}
📊 ANÁLISIS: ¿A QUÉ RAÍZ CONVERGE CADA x₀?
{'═'*80}

Valores iniciales que convergen a n = 2 (raíz doble):
"""
        for x0, r in raices_2:
            comparacion += f"   • x₀ = {x0} → {r.get('iteraciones')} iteraciones (convergencia LENTA)\n"

        if not raices_2:
            comparacion += "   (ninguno)\n"

        comparacion += f"""
Valores iniciales que convergen a n = 4 (raíz simple):
"""
        for x0, r in raices_4:
            comparacion += f"   • x₀ = {x0} → {r.get('iteraciones')} iteraciones (convergencia RÁPIDA)\n"

        if not raices_4:
            comparacion += "   (ninguno)\n"

        comparacion += f"""
{'─'*80}
💡 OBSERVACIONES CLAVE:
{'─'*80}
   • La convergencia a la raíz DOBLE n=2 es LINEAL (lenta)
   • La convergencia a la raíz SIMPLE n=4 es CUADRÁTICA (rápida)
   • Valores iniciales cercanos a n=3 pueden saltar a n=2 (T'(3)=5, pero 
     la dirección apunta hacia n=2)
   • Para garantizar convergencia a n=4, usar x₀ > 3.5

{'═'*80}
🎯 INTERPRETACIÓN DEL RESULTADO
{'═'*80}

   Las raíces representan el número óptimo de threads:
   • n = 2: Punto de inflexión (raíz doble) - NO ES ÓPTIMO
   • n = 4: Punto óptimo donde el overhead de sincronización
            equilibra el beneficio del paralelismo
   
   Recomendación: Usar n = 4 threads para este sistema

{'═'*80}
"""

        texto.insert(tk.END, comparacion)
        texto.config(state=tk.DISABLED)

        # === TAB 2: Tabla de Iteraciones Detallada ===
        frame_tabla = ttk.Frame(notebook)
        notebook.add(frame_tabla, text="📋 Iteraciones por x₀")

        # Crear sub-notebook para cada x0
        sub_notebook = ttk.Notebook(frame_tabla)
        sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for x0, resultado in resultados.items():
            frame_x0 = ttk.Frame(sub_notebook)
            sub_notebook.add(frame_x0, text=f"x₀ = {x0}")

            # Crear tabla
            tree_frame = ttk.Frame(frame_x0)
            tree_frame.pack(fill=tk.BOTH, expand=True)

            scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
            scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

            columnas = ('n', 'x_n', 'f(x_n)', 'fpx', 'x_n1', 'error_abs')
            tree = ttk.Treeview(tree_frame, columns=columnas, show='headings',
                               yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

            tree.heading('n', text='n')
            tree.heading('x_n', text='xₙ')
            tree.heading('f(x_n)', text='f(xₙ)')
            tree.heading('fpx', text="f'(xₙ)")
            tree.heading('x_n1', text='xₙ₊₁')
            tree.heading('error_abs', text='Error')

            tree.column('n', width=50, anchor=tk.CENTER)
            tree.column('x_n', width=140, anchor=tk.CENTER)
            tree.column('f(x_n)', width=120, anchor=tk.CENTER)
            tree.column('fpx', width=120, anchor=tk.CENTER)
            tree.column('x_n1', width=140, anchor=tk.CENTER)
            tree.column('error_abs', width=100, anchor=tk.CENTER)

            scroll_y.config(command=tree.yview)
            scroll_x.config(command=tree.xview)

            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)

            # Llenar tabla
            for h in resultado.get('historial', []):
                fpx_val = h.get('f_prime_xn', h.get('fpx', h.get("f'(x_n)", 0)))
                tree.insert('', tk.END, values=(
                    h['n'],
                    f"{h['x_n']:.10f}",
                    f"{h['f(x_n)']:.6e}",
                    f"{fpx_val:.6e}",
                    f"{h.get('x_n+1', h.get('x_n1', 0)):.10f}",
                    f"{h['error_absoluto']:.2e}"
                ))

        # === TAB 3: Convergencia Cuadrática ===
        frame_conv = ttk.Frame(notebook)
        notebook.add(frame_conv, text="📈 Orden de Convergencia")

        texto_conv = tk.Text(frame_conv, font=('Consolas', 11), wrap=tk.WORD)
        texto_conv.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        convergencia = f"""
{'═'*75}
ANÁLISIS DEL ORDEN DE CONVERGENCIA
{'═'*75}

Newton-Raphson tiene:
• Convergencia CUADRÁTICA (orden 2) para raíces SIMPLES
• Convergencia LINEAL (orden 1) para raíces MÚLTIPLES

Para una raíz simple:  eₙ₊₁ ≈ C · eₙ²
Para una raíz doble:   eₙ₊₁ ≈ C · eₙ

{'─'*75}
T(n) = (n-2)²(n-4):
{'─'*75}

• n = 2 es raíz DOBLE → T'(2) = 0 → Convergencia LINEAL
• n = 4 es raíz SIMPLE → T'(4) = 20 ≠ 0 → Convergencia CUADRÁTICA

Esto explica por qué:
- Convergencia a n=4 requiere pocas iteraciones (cuadrática)
- Convergencia a n=2 requiere muchas iteraciones (lineal)

{'═'*75}
"""

        texto_conv.insert(tk.END, convergencia)
        texto_conv.config(state=tk.DISABLED)

    def _mostrar_comparacion_x0(self, resultados: Dict, ejercicio: Dict):
        """Muestra ventana con comparación de diferentes valores iniciales para punto fijo."""
        ventana = tk.Toplevel(self.root)
        ventana.title("Comparación de Valores Iniciales - Punto Fijo")
        ventana.geometry("900x700")
        ventana.minsize(850, 600)

        # Notebook para organizar contenido
        notebook = ttk.Notebook(ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === TAB 1: Resumen Comparativo ===
        frame_resumen = ttk.Frame(notebook)
        notebook.add(frame_resumen, text="📊 Resumen")

        texto = tk.Text(frame_resumen, font=('Consolas', 11), wrap=tk.WORD)
        scroll = ttk.Scrollbar(frame_resumen, command=texto.yview)
        texto.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        texto.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        comparacion = f"""
{'═'*75}
   COMPARACIÓN DE VALORES INICIALES - MÉTODO DE PUNTO FIJO
   {ejercicio.get('descripcion', '')}
{'═'*75}

📐 Función: g(x) = 0.5·cos(x) + 1.5
📍 Buscamos x tal que x = g(x)
🎯 Tolerancia: {ejercicio['tolerancia']:.0e}

{'─'*75}
VERIFICACIÓN DE CONVERGENCIA:
{'─'*75}
   g'(x) = -0.5·sin(x)
   
   Para cualquier x: |g'(x)| = |−0.5·sin(x)| ≤ 0.5 < 1
   
   ✅ La condición de convergencia |g'(x)| < 1 se cumple SIEMPRE
   ✅ El método converge para CUALQUIER valor inicial x₀
{'─'*75}

{'─'*75}
{'x₀':<8} {'Iteraciones':<14} {'Raíz encontrada':<22} {'Error final':<15} {'Tiempo (ms)':<12}
{'─'*75}
"""

        # Agregar datos de cada x0
        for x0, resultado in resultados.items():
            raiz = resultado.get('raiz', 0)
            iter_count = resultado.get('iteraciones', 0)
            error = resultado.get('error_final', 0)
            tiempo = resultado.get('tiempo', 0) * 1000
            conv = "✓" if resultado.get('convergencia') else "✗"
            comparacion += f"{x0:<8.1f} {iter_count:<14d} {raiz:<22.12f} {error:<15.2e} {tiempo:<12.4f}\n"

        comparacion += f"{'─'*75}\n"

        # Análisis
        iter_min = min(r.get('iteraciones', float('inf')) for r in resultados.values())
        iter_max = max(r.get('iteraciones', 0) for r in resultados.values())
        mejor_x0 = [x0 for x0, r in resultados.items() if r.get('iteraciones') == iter_min][0]
        peor_x0 = [x0 for x0, r in resultados.items() if r.get('iteraciones') == iter_max][0]

        # Verificar que todos convergen a la misma raíz
        raices = [r.get('raiz', 0) for r in resultados.values()]
        raiz_promedio = sum(raices) / len(raices)

        comparacion += f"""
{'═'*75}
📊 ANÁLISIS: ¿CÓMO AFECTA EL VALOR INICIAL A LA CONVERGENCIA?
{'═'*75}

📍 Todos los valores iniciales convergen a la misma raíz: x ≈ {raiz_promedio:.10f}

🏆 MEJOR valor inicial:  x₀ = {mejor_x0} ({iter_min} iteraciones)
🐢 PEOR valor inicial:   x₀ = {peor_x0} ({iter_max} iteraciones)

💡 OBSERVACIONES:
   • La diferencia en iteraciones es de {iter_max - iter_min} iteraciones
   • Los valores iniciales más cercanos al punto fijo convergen más rápido
   • El punto fijo está aproximadamente en x ≈ {raiz_promedio:.4f}
   
   Distancias iniciales al punto fijo:
"""

        for x0 in sorted(resultados.keys()):
            distancia = abs(x0 - raiz_promedio)
            iters = resultados[x0].get('iteraciones', 0)
            comparacion += f"   • |x₀={x0} - raíz| = {distancia:.4f} → {iters} iteraciones\n"

        comparacion += f"""
{'═'*75}
🎯 INTERPRETACIÓN DEL RESULTADO
{'═'*75}

   El punto fijo x ≈ {raiz_promedio:.6f} representa que la base de datos SaaS
   alcanzará el 80% de su capacidad aproximadamente en:
   
   📅 {raiz_promedio:.2f} meses ≈ {int(raiz_promedio)} meses y {int((raiz_promedio % 1) * 30)} días
   
   Esto permite planificar con anticipación:
   • Expansión de almacenamiento
   • Migración a infraestructura más grande
   • Optimización de datos (archivado, limpieza)

{'═'*75}
"""

        texto.insert(tk.END, comparacion)
        texto.config(state=tk.DISABLED)

        # === TAB 2: Tabla de Iteraciones Detallada ===
        frame_tabla = ttk.Frame(notebook)
        notebook.add(frame_tabla, text="📋 Iteraciones por x₀")

        # Crear sub-notebook para cada x0
        sub_notebook = ttk.Notebook(frame_tabla)
        sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for x0, resultado in resultados.items():
            frame_x0 = ttk.Frame(sub_notebook)
            sub_notebook.add(frame_x0, text=f"x₀ = {x0}")

            # Crear tabla
            tree_frame = ttk.Frame(frame_x0)
            tree_frame.pack(fill=tk.BOTH, expand=True)

            scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
            scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

            columnas = ('n', 'x_n', 'g(x_n)', 'error_abs', 'error_rel')
            tree = ttk.Treeview(tree_frame, columns=columnas, show='headings',
                               yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

            tree.heading('n', text='n')
            tree.heading('x_n', text='xₙ')
            tree.heading('g(x_n)', text='g(xₙ)')
            tree.heading('error_abs', text='|xₙ - g(xₙ)|')
            tree.heading('error_rel', text='Error Relativo')

            tree.column('n', width=50, anchor=tk.CENTER)
            tree.column('x_n', width=150, anchor=tk.CENTER)
            tree.column('g(x_n)', width=150, anchor=tk.CENTER)
            tree.column('error_abs', width=130, anchor=tk.CENTER)
            tree.column('error_rel', width=130, anchor=tk.CENTER)

            scroll_y.config(command=tree.yview)
            scroll_x.config(command=tree.xview)

            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(fill=tk.BOTH, expand=True)

            # Llenar tabla
            for h in resultado.get('historial', []):
                tree.insert('', tk.END, values=(
                    h['n'],
                    f"{h['x_n']:.10f}",
                    f"{h['g(x_n)']:.10f}",
                    f"{h['error_absoluto']:.2e}",
                    f"{h['error_relativo']*100:.6f}%"
                ))

        # === TAB 3: Condición de Convergencia ===
        frame_conv = ttk.Frame(notebook)
        notebook.add(frame_conv, text="📐 Convergencia")

        texto_conv = tk.Text(frame_conv, font=('Consolas', 11), wrap=tk.WORD)
        texto_conv.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        condicion = f"""
{'═'*70}
ANÁLISIS DE LA CONDICIÓN DE CONVERGENCIA
{'═'*70}

Para el método de punto fijo x = g(x), la condición de convergencia es:

                    |g'(x)| < 1

Si esta condición se cumple en un intervalo que contiene el punto fijo,
entonces el método converge para cualquier x₀ en ese intervalo.

{'─'*70}
PARA g(x) = 0.5·cos(x) + 1.5:
{'─'*70}

   g'(x) = d/dx [0.5·cos(x) + 1.5]
   g'(x) = -0.5·sin(x)

   Por lo tanto:
   |g'(x)| = |-0.5·sin(x)| = 0.5·|sin(x)|

   Como |sin(x)| ≤ 1 para todo x:
   |g'(x)| = 0.5·|sin(x)| ≤ 0.5 < 1  ✅

{'─'*70}
CONCLUSIÓN:
{'─'*70}

   ✓ La condición |g'(x)| < 1 se cumple PARA TODO x
   ✓ El método SIEMPRE converge, independientemente de x₀
   ✓ El valor máximo de |g'(x)| es 0.5 (cuando |sin(x)| = 1)

   Velocidad de convergencia:
   • A menor |g'(x)|, más rápida la convergencia
   • Cerca del punto fijo, si |g'(x*)| ≈ 0.5, la convergencia es lineal
   • El error se reduce aproximadamente a la mitad en cada iteración

{'─'*70}
VERIFICACIÓN NUMÉRICA EN CADA x₀:
{'─'*70}
"""

        for x0, resultado in resultados.items():
            cond = resultado.get('condicion_g_prima')
            if cond is not None:
                condicion += f"\n   x₀ = {x0}: |g'(x₀)| ≈ {cond:.6f} {'< 1 ✓' if cond < 1 else '>= 1 ✗'}"

        condicion += f"""

{'═'*70}
"""

        texto_conv.insert(tk.END, condicion)
        texto_conv.config(state=tk.DISABLED)

    def _graficar_comparacion_x0(self, resultados: Dict, g: callable, ejercicio: Dict):
        """Grafica la comparación de convergencia para diferentes x₀."""
        # Limpiar figura de comparación
        self.fig_comparacion.clear()

        # Crear layout de 2x2
        ax1 = self.fig_comparacion.add_subplot(221)  # Convergencia de errores
        ax2 = self.fig_comparacion.add_subplot(222)  # Cobweb múltiple
        ax3 = self.fig_comparacion.add_subplot(223)  # Barras de iteraciones
        ax4 = self.fig_comparacion.add_subplot(224)  # Trayectorias

        colores = ['blue', 'red', 'green', 'purple']
        marcadores = ['o', 's', '^', 'D']

        # === Subplot 1: Convergencia del error ===
        for i, (x0, resultado) in enumerate(resultados.items()):
            historial = resultado.get('historial', [])
            if historial:
                iteraciones = [h['n'] for h in historial]
                errores = [h['error_absoluto'] for h in historial if h['error_absoluto'] > 0]
                if len(errores) == len(iteraciones):
                    ax1.semilogy(iteraciones, errores, f'-{marcadores[i]}',
                               color=colores[i], label=f'x₀ = {x0}',
                               linewidth=2, markersize=5)

        ax1.axhline(y=ejercicio['tolerancia'], color='black', linestyle='--',
                   linewidth=1.5, label=f'Tol = {ejercicio["tolerancia"]:.0e}')
        ax1.set_xlabel('Iteración', fontsize=10)
        ax1.set_ylabel('Error (log)', fontsize=10)
        ax1.set_title('Convergencia del Error', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, which='both')

        # === Subplot 2: Cobweb plot múltiple ===
        # Graficar y=x y y=g(x)
        x = np.linspace(0, 2.5, 200)
        y_identidad = x
        y_gx = [g(xi) for xi in x]

        ax2.plot(x, y_identidad, 'k-', linewidth=2, label='y = x')
        ax2.plot(x, y_gx, 'b-', linewidth=2, label='y = g(x)')

        # Dibujar cobwebs para cada x0
        for i, (x0, resultado) in enumerate(resultados.items()):
            historial = resultado.get('historial', [])
            if historial:
                # Dibujar cobweb simplificado
                xn = x0
                for j, h in enumerate(historial[:5]):  # Solo primeras 5 iteraciones
                    gxn = h['g(x_n)']
                    # Línea vertical
                    ax2.plot([xn, xn], [xn, gxn], '-', color=colores[i],
                            linewidth=1, alpha=0.6)
                    # Línea horizontal
                    ax2.plot([xn, gxn], [gxn, gxn], '-', color=colores[i],
                            linewidth=1, alpha=0.6)
                    xn = gxn

                # Marcar x0
                ax2.plot(x0, 0, f'{marcadores[i]}', color=colores[i],
                        markersize=8, label=f'x₀={x0}')

        # Marcar punto fijo
        raiz = list(resultados.values())[0].get('raiz', 0)
        if raiz:
            ax2.plot(raiz, raiz, 'g*', markersize=15, zorder=10)

        ax2.set_xlabel('x', fontsize=10)
        ax2.set_ylabel('y', fontsize=10)
        ax2.set_title('Diagrama de Telaraña (Cobweb)', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 2.5)
        ax2.set_ylim(0, 2.5)

        # === Subplot 3: Barras de iteraciones ===
        x0_valores = list(resultados.keys())
        iteraciones = [r.get('iteraciones', 0) for r in resultados.values()]

        bars = ax3.bar([f'x₀={x0}' for x0 in x0_valores], iteraciones,
                      color=colores[:len(x0_valores)])

        # Etiquetas
        for bar, iter_val in zip(bars, iteraciones):
            ax3.annotate(str(iter_val),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax3.set_ylabel('Iteraciones', fontsize=10)
        ax3.set_title('Iteraciones por Valor Inicial', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # === Subplot 4: Trayectoria de aproximaciones ===
        for i, (x0, resultado) in enumerate(resultados.items()):
            historial = resultado.get('historial', [])
            if historial:
                aprox = [x0] + [h['g(x_n)'] for h in historial]
                ax4.plot(range(len(aprox)), aprox, f'-{marcadores[i]}',
                        color=colores[i], label=f'x₀ = {x0}',
                        linewidth=1.5, markersize=4)

        # Línea del punto fijo
        if raiz:
            ax4.axhline(y=raiz, color='green', linestyle='--', linewidth=2,
                       label=f'Punto fijo ≈ {raiz:.4f}')

        ax4.set_xlabel('Iteración', fontsize=10)
        ax4.set_ylabel('Aproximación xₙ', fontsize=10)
        ax4.set_title('Trayectoria de Aproximaciones', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        self.fig_comparacion.tight_layout()
        self.canvas_comparacion.draw()

        # Cambiar a la pestaña de comparación
        self.notebook_graficas.select(3)


def main():
    """Función principal para ejecutar la aplicación."""
    root = tk.Tk()
    app = AplicacionMetodosNumericos(root)
    root.mainloop()


if __name__ == "__main__":
    main()

