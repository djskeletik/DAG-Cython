from importlib import import_module
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

try:
    lib = import_module("graph_cython.library")
except ModuleNotFoundError as e:
    raise ImportError(
        "Cython extension 'graph_cython.library' не найден. Сначала собери библиотеку:\n"
        "  cd graph_cython && python setup.py build_ext --inplace"
    ) from e

# Re-export
Input = getattr(lib, "Input")
Node = getattr(lib, "Node")
Sum = getattr(lib, "Sum")
SumDouble = getattr(lib, "SumDouble")
Product = getattr(lib, "Product")
MatrixProduct = getattr(lib, "MatrixProduct")
Sin = getattr(lib, "Sin")

__all__ = ["Input", "Node", "Sum", "SumDouble", "Product", "MatrixProduct", "Sin"]
