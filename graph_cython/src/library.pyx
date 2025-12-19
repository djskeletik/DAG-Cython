# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Ядро DAG на Cython для вычислений с массивами numpy.

Классы:
- Node: базовый узел DAG.
- Input: узел для входных данных.
- Sum, SumDouble: сложение элементов.
- Product: поэлементное умножение.
- MatrixProduct: умножение матриц.
- Sin: поэлементный синус.

Особенности:
- Итеративное построение топологического порядка узлов (to_c_struct)
- Итеративное выполнение DAG (run)
- Использует numpy-массивы и memoryview для быстродействия
"""

from cpython.ref cimport Py_INCREF, Py_CLEAR
import numpy as np
cimport numpy as cnp

ctypedef double DTYPE_t

# Инициализация C-API numpy
cnp.import_array()

# ---------------- Класс Node ----------------
cdef class Node:
    """
    Базовый узел DAG.

    Свойства:
    - inputs: список входных узлов
    - outputs: список выходных узлов
    - _order: топологический порядок узлов
    - _dirty: флаг, указывающий, что порядок нужно пересчитать
    - _last: результат последнего вычисления узла
    """
    cdef public object inputs, outputs, _order, _last
    cdef public bint _dirty

    def __cinit__(self):
        self.inputs = []
        self.outputs = []
        self._order = None
        self._dirty = True
        self._last = None

    def __rshift__(self, other):
        """
        Позволяет связать узлы DAG через оператор >>.
        self >> other добавляет self как вход для other.
        """
        if not isinstance(other, Node):
            raise TypeError("Правый операнд должен быть Node")
        self.outputs.append(other)
        other.inputs.append(self)
        other._dirty = True
        return other

    cpdef object to_c_struct(self):
        """
        Построение топологического порядка узлов DAG.
        Возвращает список узлов в порядке вычисления.
        """
        if not self._dirty and self._order is not None:
            return self._order

        cdef list stack = [self]
        cdef dict seen = {}
        cdef list all_nodes = []
        cdef Node cur
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen[id(cur)] = cur
            all_nodes.append(cur)
            for inp in cur.inputs:
                if id(inp) not in seen:
                    stack.append(inp)

        cdef dict indeg = {}
        for cur in all_nodes:
            indeg[id(cur)] = 0
        for cur in all_nodes:
            for out in cur.outputs:
                if id(out) in indeg:
                    indeg[id(out)] += 1

        cdef list q = []
        for cur in all_nodes:
            if indeg[id(cur)] == 0:
                q.append(cur)

        cdef list order = []
        cdef Node n, m
        while q:
            n = q.pop()
            order.append(n)
            for m in n.outputs:
                if id(m) in indeg:
                    indeg[id(m)] -= 1
                    if indeg[id(m)] == 0:
                        q.append(m)

        self._order = order
        self._dirty = False
        return order

    cpdef object run(self):
        """
        Вычисление DAG в топологическом порядке.
        """
        cdef list order = self.to_c_struct()
        for node in order:
            node._last = None
        for node in order:
            if node._last is None:
                node._last = node.compute()
        return self._last

    cpdef object compute(self):
        """
        Метод, который должны реализовать наследники.
        """
        raise NotImplementedError("Подклассы должны реализовать compute()")

# ---------------- Класс Input ----------------
cdef class Input(Node):
    """
    Узел для входных данных.
    """
    cdef public object _arr

    def __init__(self, data):
        self._arr = np.asarray(data, dtype=np.float64)
        self._dirty = False

    cpdef object compute(self):
        """
        Возвращает входной массив.
        """
        self._last = self._arr
        return self._last

# ---------------- Класс Sum ----------------
cdef class Sum(Node):
    """
    Суммирование нескольких входов.
    """
    def __init__(self, *args):
        for a in args:
            if isinstance(a, Node):
                a >> self
            else:
                Input(a) >> self

    cpdef object compute(self):
        """
        Выполняет суммирование всех входов.
        """
        if not self.inputs:
            self._last = np.array([], dtype=np.float64)
            return self._last
        res = np.array(self.inputs[0]._last if self.inputs[0]._last is not None else self.inputs[0].compute(),
                       copy=True)
        cdef Py_ssize_t i
        for i in range(1, len(self.inputs)):
            b = self.inputs[i]._last if self.inputs[i]._last is not None else self.inputs[i].compute()
            res += b
        self._last = res
        return res

# ---------------- Класс SumDouble ----------------
cdef class SumDouble(Node):
    """
    Узел для сложения двух входов.
    Если входов больше двух, использует Sum.
    """
    def __init__(self, a=None, b=None):
        if a is not None:
            if isinstance(a, Node):
                a >> self
            else:
                Input(a) >> self
        if b is not None:
            if isinstance(b, Node):
                b >> self
            else:
                Input(b) >> self

    cpdef object compute(self):
        """
        Вычисляет сумму двух входов.
        """
        if len(self.inputs) != 2:
            tmp = Sum(*self.inputs)
            return tmp.compute()
        a_node, b_node = self.inputs[0], self.inputs[1]
        a = a_node._last if a_node._last is not None else a_node.compute()
        b = b_node._last if b_node._last is not None else b_node.compute()
        self._last = a + b
        return self._last

# ---------------- Класс Product ----------------
cdef class Product(Node):
    """
    Поэлементное умножение нескольких входов.
    """
    def __init__(self, *args):
        for a in args:
            if isinstance(a, Node):
                a >> self
            else:
                Input(a) >> self

    cpdef object compute(self):
        """
        Вычисляет произведение всех входов.
        """
        if not self.inputs:
            self._last = np.array([], dtype=np.float64)
            return self._last
        res = np.array(self.inputs[0]._last if self.inputs[0]._last is not None else self.inputs[0].compute(),
                       copy=True)
        cdef Py_ssize_t i
        for i in range(1, len(self.inputs)):
            res *= self.inputs[i]._last if self.inputs[i]._last is not None else self.inputs[i].compute()
        self._last = res
        return self._last

# ---------------- Класс MatrixProduct ----------------
cdef class MatrixProduct(Node):
    """
    Умножение двух матриц.
    """
    def __init__(self, a=None, b=None):
        if a is not None:
            if isinstance(a, Node):
                a >> self
            else:
                Input(a) >> self
        if b is not None:
            if isinstance(b, Node):
                b >> self
            else:
                Input(b) >> self

    cpdef object compute(self):
        """
        Вычисляет произведение двух матриц.
        """
        if len(self.inputs) != 2:
            raise ValueError("MatrixProduct требует ровно 2 входа")
        a_node, b_node = self.inputs[0], self.inputs[1]
        a = a_node._last if a_node._last is not None else a_node.compute()
        b = b_node._last if b_node._last is not None else b_node.compute()
        self._last = a.dot(b)
        return self._last

# ---------------- Класс Sin ----------------
cdef class Sin(Node):
    """
    Поэлементный синус входного массива.
    """
    def __init__(self, a=None):
        if a is not None:
            if isinstance(a, Node):
                a >> self
            else:
                Input(a) >> self

    cpdef object compute(self):
        """
        Вычисляет sin для всех элементов входного массива.
        """
        if not self.inputs:
            self._last = np.array([], dtype=np.float64)
            return self._last
        val = self.inputs[0]._last if self.inputs[0]._last is not None else self.inputs[0].compute()
        self._last = np.sin(val)
        return self._last
