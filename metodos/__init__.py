"""
Módulo de métodos numéricos para resolver ecuaciones no lineales.
"""

from .biseccion import MetodoBiseccion
from .falsa_posicion import MetodoFalsaPosicion
from .punto_fijo import MetodoPuntoFijo
from .newton import MetodoNewton
from .secante import MetodoSecante

__all__ = [
    'MetodoBiseccion',
    'MetodoFalsaPosicion',
    'MetodoPuntoFijo',
    'MetodoNewton',
    'MetodoSecante'
]

