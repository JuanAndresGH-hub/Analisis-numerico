# 📊 Métodos Numéricos - Resolución de Ecuaciones No Lineales

Aplicación con interfaz gráfica en Python para resolver ecuaciones no lineales mediante cinco métodos numéricos clásicos.

## 📋 Descripción

Este proyecto implementa una interfaz gráfica completa utilizando Tkinter para la resolución de ecuaciones no lineales f(x) = 0. Incluye visualización de gráficas con Matplotlib y análisis detallado de convergencia.

## 🔧 Métodos Implementados

| Método | Descripción | Tipo de Convergencia |
|--------|-------------|---------------------|
| **Bisección** | Divide el intervalo a la mitad repetidamente | Lineal |
| **Falsa Posición** | Interpolación lineal entre extremos | Superlineal |
| **Punto Fijo** | Iteración x_{n+1} = g(x_n) | Lineal |
| **Newton-Raphson** | Usa tangentes: x_{n+1} = x_n - f(x_n)/f'(x_n) | Cuadrática |
| **Secante** | Similar a Newton pero sin derivada | Superlineal (~1.618) |

## 📦 Requisitos

- Python 3.8 o superior
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SymPy >= 1.10.0
- Tkinter (incluido en Python estándar)

## 🚀 Instalación

1. Clonar o descargar el repositorio:
```bash
git clone <url-del-repositorio>
cd "Analisis numerico"
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Ejecutar la aplicación

```bash
python main.py
```

### Ejecutar pruebas

```bash
python -m pytest tests/ -v
# o
python -m unittest tests.test_metodos -v
```

## 📁 Estructura del Proyecto

```
proyecto_metodos_numericos/
│
├── main.py                    # Punto de entrada de la aplicación
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
│
├── metodos/                   # Implementación de métodos numéricos
│   ├── __init__.py
│   ├── biseccion.py          # Método de Bisección
│   ├── falsa_posicion.py     # Método de Falsa Posición
│   ├── punto_fijo.py         # Método de Punto Fijo
│   ├── newton.py             # Método de Newton-Raphson
│   └── secante.py            # Método de la Secante
│
├── interfaz/                  # Componentes de la GUI
│   ├── __init__.py
│   └── gui_principal.py      # Ventana principal
│
├── funciones/                 # Definiciones de funciones
│   ├── __init__.py
│   └── definiciones.py       # Funciones de los ejercicios
│
├── utils/                     # Utilidades
│   ├── __init__.py
│   └── validaciones.py       # Funciones de validación
│
└── tests/                     # Pruebas unitarias
    ├── __init__.py
    └── test_metodos.py       # Tests de los métodos
```

## 📝 Ejercicios Incluidos

### Ejercicio 1 - Bisección
- **Función:** T(λ) = 2.5 + 0.8λ² - 3.2λ + ln(λ + 1)
- **Intervalo:** [0.5, 2.5]
- **Tolerancia:** 1e-6
- **Nota:** Esta función no tiene raíces reales en el intervalo dado. El método detectará correctamente que no hay cambio de signo.

### Ejercicio 2 - Falsa Posición
- **Función:** E(x) = x³ - 6x² + 11x - 6.5
- **Intervalo:** [2, 4]
- **Tolerancia:** 1e-7

### Ejercicio 3 - Punto Fijo
- **Ecuación:** x = 0.5cos(x) + 1.5
- **Valor inicial:** x₀ = 1.0
- **Tolerancia:** 1e-8

### Ejercicio 4 - Newton-Raphson
- **Función:** T(n) = n³ - 8n² + 20n - 16
- **Valor inicial:** x₀ = 3.0
- **Tolerancia:** 1e-10

### Ejercicio 5 - Secante
- **Función:** P(x) = x·e^(-x/2) - 0.3
- **Valores iniciales:** x₀ = 0.5, x₁ = 1.0
- **Tolerancia:** 1e-9

## 🎨 Características de la Interfaz

### Panel de Entrada
- Selección de método numérico
- Selección de ejercicio predefinido
- Campos para función, intervalos, valores iniciales
- Tolerancia y máximo de iteraciones configurables

### Tabla de Resultados
- Muestra todas las iteraciones con columnas personalizadas por método
- **Formato de 8 decimales** para precisión
- **Error relativo en porcentaje (%)** para fácil interpretación
- Notación científica para valores muy pequeños o grandes

### Gráficas (4 pestañas)
1. **Gráfica de la Función:** Muestra f(x), las aproximaciones y la raíz final
2. **Gráfica de Convergencia:** 
   - Error absoluto y relativo vs iteración en **escala logarítmica**
   - Línea de tolerancia
   - **Análisis de convergencia cuadrática** para Newton-Raphson
   - Anotación del error final
3. **Gráficas Especiales:**
   - **Newton:** Tangentes dinámicas con colores degradados en cada iteración
   - **Secante:** Rectas secantes con visualización completa
   - **Punto Fijo:** **Cobweb plot completo** con condición de convergencia |g'(x)|
   - **Bisección/Falsa Posición:** Reducción del intervalo como barras
4. **Comparación Newton vs Secante:**
   - Gráfica comparativa de convergencia
   - Barras comparativas de iteraciones y evaluaciones

### Panel de Resultados Mejorado
- Raíz aproximada con **12 decimales**
- Número de iteraciones
- **Error absoluto y relativo**
- Tiempo de ejecución en milisegundos
- Evaluaciones de función
- **Mensaje claro de CONVERGENCIA/FALLA** con explicación
- Información adicional por método (orden de convergencia, evaluaciones de derivada)

## 📊 Comparación de Métodos

La aplicación incluye **dos botones de comparación**:

### Botón "Comparar Bis/FP"
Compara bisección y falsa posición en el mismo problema:
- Tabla comparativa detallada
- Gráfica superpuesta de convergencia en escala logarítmica
- Análisis de cuál método converge más rápido y por qué

### Botón "Newton vs Secante"
Compara Newton-Raphson y Secante:
- Tabla con todas las métricas (iteraciones, evaluaciones, tiempo)
- **Gráfica doble:** convergencia del error + barras comparativas
- Análisis del costo de calcular derivadas
- Recomendación basada en la complejidad del problema

## ⚠️ Manejo de Errores

El programa maneja los siguientes casos:
- División por cero
- No convergencia
- Intervalo inválido (sin cambio de signo)
- Condiciones de convergencia no satisfechas
- Valores NaN o infinitos

## 🔍 Validaciones

- Verificación de cambio de signo para métodos de intervalo
- Verificación de |g'(x)| < 1 para punto fijo
- Verificación de f'(x) ≠ 0 para Newton-Raphson
- Verificación de denominador no cero para secante

## 📚 Referencias

- Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.)
- Chapra, S. C., & Canale, R. P. (2015). Numerical Methods for Engineers
- Press, W. H., et al. (2007). Numerical Recipes: The Art of Scientific Computing

## 📄 Licencia

Este proyecto es de uso académico para el curso de Análisis Numérico.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue primero para discutir los cambios que desea realizar.

# Analisis-numerico
# Analisis-numerico
