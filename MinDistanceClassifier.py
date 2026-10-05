from typing import List, Tuple


class MinDistanceClassifier:
    """Классификатор на основе минимального расстояния до прототипа класса."""

    def __init__(self):
        self.prototypes: List[Tuple[float, ...]] = []
        self.classes: List[int] = []
        self.dim: int = 0
        self._trained = False

    def fit(self, train_set: List[Tuple[float, ...]]):
        """прототип каждого класса как среднее арифметическое точек."""
        groups = {}
        for sample in train_set:
            *features, c = sample
            groups.setdefault(c, []).append(tuple(features))

        self.dim = len(next(iter(groups.values()))[0])
        self.classes = sorted(groups.keys())
        self.prototypes = []

        for c in self.classes:
            pts = groups[c]
            n = len(pts)
            # Среднее по каждой координате.
            proto = tuple(
                sum(p[j] for p in pts) / n
                for j in range(self.dim)
            )
            self.prototypes.append(proto)

        self._trained = True
        return self

    def decision_values(self, *x) -> List[float]:
        """Решающая функция D_k = 2·(x·p_k) − |p_k|²."""
        if not self._trained:
            raise RuntimeError("Модель не обучена")
        result = []
        for proto in self.prototypes:
            dot = sum(xi * pi for xi, pi in zip(x, proto))
            norm_sq = sum(pi * pi for pi in proto)
            result.append(2 * dot - norm_sq)
        return result

    def predict(self, *x) -> int:
        """ макс. значение решающей функции."""
        values = self.decision_values(*x)
        idx = max(range(len(values)), key=lambda i: values[i])
        return self.classes[idx]

    def decision_equation(self, i: int) -> str:
        proto = self.prototypes[i]
        terms = " + ".join(f"{2*p:.3f}·X{j+1}" for j, p in enumerate(proto))
        c = -sum(p * p for p in proto)
        return f"D{i+1} = {terms} {c:+.3f}"

    def decision_between_value(self, i: int, j: int, *x) -> float:
        """Значение разделяющей границы между классами i и j в точке x."""
        p = self.prototypes[i]
        q = self.prototypes[j]
        dot = sum(xi * (pi - qi) for xi, pi, qi in zip(x, p, q))
        return 2 * dot + sum(qi * qi - pi * pi for pi, qi in zip(p, q))