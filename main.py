import os
import subprocess
import numpy as np
from graph_cython import Input, SumDouble
from tests.test import (
    example_basic_arithmetic,
    example_matrix_operations,
    example_trigonometric,
    performance_test
)
from tests.test_advanced import make_advanced_graph, run_advanced_test

# Настройки графа
WIDTH = 8
LENGTH = 6
DSIZE = 2
RUNS = 5
ARRAY_SIZE = 10000

# Чтобы массивы печатались красиво, без e+XX
np.set_printoptions(suppress=True, precision=5)

print("[LOG] === Запуск main.py ===")

# ======================== Сборка Cython ==========================
so_file = "graph_cython/library.cpython-311-darwin.so"
if os.path.exists(so_file):
    os.remove(so_file)
    print("[LOG] Старый .so файл удалён ✅")
else:
    print("[LOG] Старого .so файла нет, продолжаем ✅")

print("[LOG] Запускаем сборку Cython модуля...")
try:
    subprocess.check_call(["python3", "graph_cython/setup.py", "build_ext", "--inplace"])
    print("[LOG] Cython сборка завершена ✅")
except Exception as e:
    print(f"[LOG] Ошибка сборки ❌: {e}")

# ======================== Импорт модуля ==========================
try:
    import graph_cython
    print("[LOG] graph_cython успешно импортирован ✅")
except Exception as e:
    print(f"[LOG] Ошибка импорта graph_cython ❌: {e}")

# ======================== Базовые тесты =========================
print("\n[LOG] === Запуск базовых тестов ===")
try:
    example_basic_arithmetic()  # исправлено внутри: (a+b)*c = [10,14,18]
    example_matrix_operations()
    example_trigonometric()
except Exception as e:
    print(f"[LOG] Ошибка при запуске базовых тестов ❌: {e}")

# ======================== Продвинутый тест ======================
print("\n[LOG] === Запуск продвинутого теста ===")
try:
    nsums, data_node, head = make_advanced_graph(datasize=DSIZE, width=WIDTH, length=LENGTH)
    print(f"Создано узлов: {nsums}")
    run_advanced_test(head, runs=RUNS)
except Exception as e:
    print(f"[LOG] Ошибка при запуске продвинутого теста ❌: {e}")

# ======================== Тест производительности =================
print("\n[LOG] === Запуск теста производительности ===")
try:
    avg_time, result = performance_test(array_size=ARRAY_SIZE, width=WIDTH, depth=LENGTH, runs=RUNS)
    print(f"Среднее время выполнения: {avg_time:.5f} секунд ({avg_time:.5f})")
    print(f"Первые 5 элементов результата: {result[:5]}")
    print(f"Последние 5 элементов результата: {result[-5:]}")
except Exception as e:
    print(f"[LOG] Ошибка при запуске теста производительности ❌: {e}")
