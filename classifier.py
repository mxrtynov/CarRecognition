from typing import List, Tuple


class MinDistanceClassifier:
    def __init__(self):
        self.prototypes: List[Tuple[float, float]] = []
        self.classes: List[int] = []
        self._trained = False

    def fit(self, train_set: List[Tuple[float, float, int]]):
        groups = {}
        for x1, x2, c in train_set:
            groups.setdefault(c, []).append((x1, x2))

        self.classes = sorted(groups.keys())
        self.prototypes = []
        for c in self.classes:
            pts = groups[c]
            n = len(pts)
            px1 = sum(p[0] for p in pts) / n
            px2 = sum(p[1] for p in pts) / n
            self.prototypes.append((px1, px2))

        self._trained = True
        return self

    def decision_values(self, x1: float, x2: float) -> List[float]:
        if not self._trained:
            raise RuntimeError("Модель не обучена")
        result = []
        for (p1, p2) in self.prototypes:
            d = 2 * (x1 * p1 + x2 * p2) - (p1 * p1 + p2 * p2)
            result.append(d)
        return result

    def predict(self, x1: float, x2: float) -> int:
        values = self.decision_values(x1, x2)
        idx = max(range(len(values)), key=lambda i: values[i])
        return self.classes[idx]

    def decision_equation(self, i: int) -> str:
        p1, p2 = self.prototypes[i]
        a = 2 * p1
        b = 2 * p2
        c = -(p1 * p1 + p2 * p2)
        return f"D{i+1} = {a:.3f}·X1 + {b:.3f}·X2 {c:+.3f}"

    def decision_between(self, i: int, j: int) -> str:
        p1, p2 = self.prototypes[i]
        q1, q2 = self.prototypes[j]
        a = 2 * (p1 - q1)
        b = 2 * (p2 - q2)
        c = (q1 * q1 + q2 * q2) - (p1 * p1 + p2 * p2)
        return f"D{i+1}{j+1} = {a:+.3f}·X1 {b:+.3f}·X2 {c:+.3f}"

    def decision_between_value(self, i: int, j: int,
                               x1: float, x2: float) -> float:
        p1, p2 = self.prototypes[i]
        q1, q2 = self.prototypes[j]
        return (2 * (x1 * (p1 - q1) + x2 * (p2 - q2))
                + (q1 * q1 + q2 * q2) - (p1 * p1 + p2 * p2))


class PerceptronClassifier:
    def __init__(self, max_iter: int = 10000):
        self.w = [0.0, 0.0, 0.0]
        self.max_iter = max_iter
        self.history = []
        self.classes: List[int] = []
        self._trained = False
        self._converged = False
        self._epochs = 0
        self._updates = 0

    def fit(self, train_set: List[Tuple[float, float, int]]):
        classes = sorted({c for _, _, c in train_set})
        if len(classes) != 2:
            raise ValueError("Алгоритм восприятия работает только с двумя классами")
        self.classes = classes
        c1, c2 = classes[0], classes[1]

        data = []
        for x1, x2, c in train_set:
            sign = 1 if c == c1 else -1
            data.append((x1, x2, sign))

        self.w = [0.0, 0.0, 0.0]
        self.history = []
        self._epochs = 0
        self._updates = 0
        self._converged = False

        n = len(data)
        e = 0
        idx = 0

        for it in range(self.max_iter):
            if idx == 0:
                self._epochs += 1

            x1, x2, sign = data[idx]
            d = self.w[0] * x1 + self.w[1] * x2 + self.w[2]

            correct = (sign > 0 and d > 0) or (sign < 0 and d < 0)

            if correct:
                e += 1
                if e == n:
                    self._converged = True
                    break
            else:
                if sign > 0:
                    self.w[0] += x1
                    self.w[1] += x2
                    self.w[2] += 1.0
                else:
                    self.w[0] -= x1
                    self.w[1] -= x2
                    self.w[2] -= 1.0

                self.history.append((x1, x2, sign, self.w.copy()))
                self._updates += 1
                e = 0

            idx = (idx + 1) % n

        self._trained = True
        return self

    def decision_value(self, x1: float, x2: float) -> float:
        if not self._trained:
            raise RuntimeError("Модель не обучена")
        return self.w[0] * x1 + self.w[1] * x2 + self.w[2]

    def predict(self, x1: float, x2: float) -> int:
        return self.classes[0] if self.decision_value(x1, x2) > 0 else self.classes[1]

    def decision_equation(self) -> str:
        return (f"D12 = {self.w[0]:.3f}·X1 {self.w[1]:+.3f}·X2 "
                f"{self.w[2]:+.3f}")

    def boundary_points(self, x1_min=0.0, x1_max=1.0):
        w1, w2, w0 = self.w
        if abs(w2) > 1e-9:
            x2a = -(w1 * x1_min + w0) / w2
            x2b = -(w1 * x1_max + w0) / w2
            return x1_min, x2a, x1_max, x2b
        elif abs(w1) > 1e-9:
            x1 = -w0 / w1
            return x1, 0.0, x1, 1.0
        else:
            return 0.0, 0.0, 1.0, 0.0