import numpy as np
import timeit
from graph_cython import Input, SumDouble

np.random.seed(42)

def make_advanced_graph(datasize=1, width=6, length=5):
    """
    Создаёт продвинутый DAG граф с заданной шириной и длиной.

    Параметры:
    - datasize: размер входного массива
    - width: количество веток на каждом уровне
    - length: количество уровней

    Возвращает:
    - nsums: количество созданных узлов SumDouble
    - data_node: входной узел с данными
    - head: конечный узел графа
    """
    nsums = 0
    prevlayer = []

    # Создание входных данных
    data = np.random.uniform(-100, 100, size=datasize)
    data_node = Input(data)

    # Построение графа
    for ilayer in reversed(range(length)):
        ilayer_next = ilayer - 1
        n_groups = int(width ** ilayer_next) if ilayer_next >= 0 else 1
        thislayer = []

        for _ in range(n_groups):
            head = SumDouble()
            nsums += 1

            if prevlayer:
                for node in prevlayer:
                    node >> head
            else:
                for _ in range(width):
                    data_node >> head

            thislayer.append(head)

        prevlayer = thislayer

    return nsums, data_node, head


def run_advanced_test(head, runs=10):
    """
    Выполняет DAG и измеряет среднее время выполнения.

    Параметры:
    - head: конечный узел графа
    - runs: количество прогонов
    """
    def test():
        head.run()

    avg_time = timeit.timeit(test, number=runs) / runs
    # выводим время в обычной десятичной форме с 5 знаками после запятой
    print(f"[Advanced test] Среднее время выполнения за {runs} прогонов: {avg_time:.5f} секунд")
