import time
import random
from typing import List, Tuple, Dict

from MinDistanceClassifier import MinDistanceClassifier
from PerceptronClassifier import PerceptronClassifier


def theoretical(dim: int = 5) -> str:
    return (
        "ОБОЗНАЧЕНИЯ\n"
        f"  N  — число объектов обучающего множества\n"
        f"  d  — размерность признакового пространства (d = {dim})\n"
        "  K  — число классов (K = 2)\n"
        "  I  — число эпох обучения перцептрона\n\n"
        "МЕТОД МИНИМАЛЬНОГО РАССТОЯНИЯ\n"
        "  Обучение       : O(N·d) — один проход, суммирование по классам\n"
        "  Распознавание  : O(K·d) — вычисление K решающих функций\n"
        "  Память         : O(K·d) — хранение K прототипов\n\n"
        "АЛГОРИТМ ВОСПРИЯТИЯ\n"
        "  Обучение       : O(I·N·d) — до I эпох, каждая эпоха по N объектам\n"
        "  Распознавание  : O(d) — одно скалярное произведение\n"
        "  Память         : O(d) — один вектор весов\n\n"
        "ВЫВОД\n"
        "  Обучение: метод минимального расстояния строго быстрее,\n"
        "           т.к. выполняется за один проход (I ≥ 1).\n"
        "  Распознавание: восприятие в K раз быстрее при K классах,\n"
        "           но требует линейной разделимости классов.\n"
    )


def measure(train_set, repeats: int = 200) -> Dict[str, float]:
    result = {}

    t0 = time.perf_counter()
    for _ in range(repeats):
        MinDistanceClassifier().fit(train_set)
    result["md_fit"] = (time.perf_counter() - t0) / repeats

    md = MinDistanceClassifier().fit(train_set)
    sample = train_set[0][:-1]  # признаки без метки класса
    t0 = time.perf_counter()
    for _ in range(repeats):
        md.predict(*sample)
    result["md_predict"] = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        PerceptronClassifier().fit(train_set)
    result["pc_fit"] = (time.perf_counter() - t0) / repeats

    pc = PerceptronClassifier().fit(train_set)
    t0 = time.perf_counter()
    for _ in range(repeats):
        pc.predict(*sample)
    result["pc_predict"] = (time.perf_counter() - t0) / repeats

    return result


def report(train_set) -> str:
    m = measure(train_set)
    dim = len(train_set[0]) - 1
    lines = [theoretical(dim), "ЭКСПЕРИМЕНТ (среднее по 200 запускам)", ""]
    lines.append(f"  Обучение, мин. расстояние : {m['md_fit']*1e6:9.3f} мкс")
    lines.append(f"  Обучение, восприятие      : {m['pc_fit']*1e6:9.3f} мкс")
    lines.append(f"  Распознавание, мин. расст.: {m['md_predict']*1e6:9.3f} мкс")
    lines.append(f"  Распознавание, восприятие : {m['pc_predict']*1e6:9.3f} мкс")
    return "\n".join(lines)


def _make_synthetic(n: int, dim: int = 5):
    data = []
    for _ in range(n):
        c = random.choice([1, 2])
        if c == 1:
            features = tuple(random.uniform(0.05, 0.45) for _ in range(dim))
        else:
            features = tuple(random.uniform(0.55, 0.95) for _ in range(dim))
        data.append((*features, c))
    return data


def measure_scaling(sizes: List[int] = None,
                    dim: int = 5,
                    repeats: int = 20) -> Dict[str, List[float]]:
    if sizes is None:
        sizes = [100, 200, 500, 1000, 2000, 5000]

    result = {
        "sizes": sizes,
        "md_fit": [],
        "pc_fit": [],
        "md_predict": [],
        "pc_predict": [],
    }

    for n in sizes:
        train = _make_synthetic(n, dim=dim)

        t0 = time.perf_counter()
        for _ in range(repeats):
            MinDistanceClassifier().fit(train)
        result["md_fit"].append((time.perf_counter() - t0) / repeats)

        t0 = time.perf_counter()
        for _ in range(repeats):
            PerceptronClassifier().fit(train)
        result["pc_fit"].append((time.perf_counter() - t0) / repeats)

        md = MinDistanceClassifier().fit(train)
        pc = PerceptronClassifier().fit(train)
        sample = train[0][:-1]

        t0 = time.perf_counter()
        for _ in range(repeats):
            md.predict(*sample)
        result["md_predict"].append((time.perf_counter() - t0) / repeats)

        t0 = time.perf_counter()
        for _ in range(repeats):
            pc.predict(*sample)
        result["pc_predict"].append((time.perf_counter() - t0) / repeats)

    return result


def scaling_report(sizes: List[int] = None, dim: int = 5) -> str:
    data = measure_scaling(sizes=sizes, dim=dim)
    lines = [f"ЗАВИСИМОСТЬ ВРЕМЕНИ ОТ РАЗМЕРА ВЫБОРКИ N (d = {dim})", ""]
    header = (f"{'N':>6} | {'MD fit, мкс':>12} | {'PC fit, мкс':>12} | "
              f"{'MD pred, мкс':>13} | {'PC pred, мкс':>13}")
    lines.append(header)
    lines.append("-" * len(header))
    for i, n in enumerate(data["sizes"]):
        lines.append(
            f"{n:>6} | {data['md_fit'][i]*1e6:>12.3f} | "
            f"{data['pc_fit'][i]*1e6:>12.3f} | "
            f"{data['md_predict'][i]*1e6:>13.3f} | "
            f"{data['pc_predict'][i]*1e6:>13.3f}"
        )
    lines.append("")
    lines.append("Ожидаемый рост:")
    lines.append("  MD fit       ~ линейно по N  (O(N·d))")
    lines.append("  PC fit       ~ линейно по N·I (O(I·N·d))")
    lines.append("  MD predict   ~ константа (O(K·d))")
    lines.append("  PC predict   ~ константа (O(d))")
    return "\n".join(lines)