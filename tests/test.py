import numpy as np
import timeit
from graph_cython import Input, SumDouble, Product, MatrixProduct, Sin

# ---------------- Настройки отображения ----------------
# Чтобы массивы печатались красиво, без научной нотации e+XX, с 5 знаками после запятой
np.set_printoptions(suppress=True, precision=5)
np.random.seed(42)  # фиксируем генератор случайных чисел для воспроизводимости


# ---------------- Базовые арифметические операции ----------------
def example_basic_arithmetic():
    print("=== Базовые арифметические операции ===")

    # Создаём входные данные
    a = Input(np.array([1, 2, 3]))
    b = Input(np.array([4, 5, 6]))
    c = Input(np.array([2, 2, 2]))  # Исправлено для правильного сложения

    # Строим граф: сначала суммируем a и b, потом умножаем на c
    sum_node = SumDouble(a, b)
    product_node = Product(sum_node, c)

    # Запускаем вычисление
    result = product_node.run()
    print("(a + b) * c =", result)
    print("Ожидаемый результат: [10, 14, 18]")  # проверка корректности


# ---------------- Операции с матрицами ----------------
def example_matrix_operations():
    print("=== Операции с матрицами ===")

    # Создаём входные матрицы
    A = Input(np.array([[1, 2], [3, 4]]))
    B = Input(np.array([[5, 6], [7, 8]]))

    # Вычисляем произведение матриц
    matprod = MatrixProduct(A, B)
    result = matprod.run()

    print("A * B =", result)
    print("Ожидаемый результат: [[19, 22], [43, 50]]")  # проверка корректности


# ---------------- Тригонометрические функции ----------------
def example_trigonometric():
    print("=== Тригонометрические функции ===")

    # Входные углы
    angles = Input(np.array([0, np.pi / 2, 1e-6]))

    # Вычисляем sin
    sin_node = Sin(angles)
    result = sin_node.run()

    print("sin(angles) =", result)
    print("Ожидаемый результат: [0.0, 1.0, 0.0] (приблизительно)")


# ---------------- Тест производительности ----------------
def performance_test(array_size=10000, width=4, depth=5, runs=10):
    """
    Создаёт граф вычислений с заданной глубиной и шириной и измеряет среднее время выполнения.
    :param array_size: размер входного массива
    :param width: ширина графа (число ветвей на слой)
    :param depth: глубина графа (число слоёв)
    :param runs: количество прогонов для усреднения времени
    """
    print("=== Тест производительности ===")

    prevlayer = []  # предыдущий слой узлов
    data_node = Input(np.arange(array_size))  # входные данные
    n_nodes = 0  # счётчик всех узлов графа

    # Строим граф по слоям, начиная с нижнего (глубина depth)
    for ilayer in reversed(range(depth)):
        thislayer = []
        n_groups = max(int(width ** max(ilayer - 1, 0)), 1)  # число узлов в слое

        for _ in range(n_groups):
            node = SumDouble()  # создаём узел суммы
            n_nodes += 1

            # Соединяем с предыдущим слоем
            if prevlayer:
                for p in prevlayer:
                    p >> node
            else:
                # Если слой нижний, подключаем данные
                for _ in range(width):
                    data_node >> node
            thislayer.append(node)

        prevlayer = thislayer  # текущий слой становится предыдущим

    head = prevlayer[0]  # "голова" графа (выходной узел)
    head.to_c_struct()  # формируем топологический порядок для ускорения вычислений

    print(f"Создано узлов: {n_nodes}")

    # Функция для измерения времени выполнения
    def test():
        head.run()

    avg_time = timeit.timeit(test, number=runs) / runs  # среднее время за несколько прогонов
    result = head.run()  # финальный результат вычислений

    # Выводим результаты
    print(f"Среднее время выполнения для массивов размером {array_size}: {avg_time:.5f} секунд")
    print(f"Первые 5 элементов результата: {result[:5]}")
    print(f"Последние 5 элементов результата: {result[-5:]}")

    return avg_time, result
