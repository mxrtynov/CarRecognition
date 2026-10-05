from typing import List, Tuple


class PerceptronClassifier:
    """алгоритм восприятия."""
    def __init__(self, max_iter: int = 10000):
        self.w: List[float] = []
        self.max_iter = max_iter
        self.history = []
        self.classes: List[int] = []
        self.dim: int = 0
        self._trained = False
        self._converged = False
        self._epochs = 0
        self._updates = 0

    def fit(self, train_set: List[Tuple[float, ...]]):
        """коррекция весов до сходимости."""
        classes = sorted({sample[-1] for sample in train_set})
        if len(classes) != 2:
            raise ValueError("Алгоритм восприятия работает только с двумя классами")
        self.classes = classes


        c1, c2 = classes[0], classes[1]
        data = []
        for sample in train_set:
            *features, c = sample
            sign = 1 if c == c1 else -1
            data.append((tuple(features), sign))



        self.dim = len(data[0][0])
        # Вектор весов: d весов + смещение w0.
        self.w = [0.0] * (self.dim + 1)
        self.history = []
        self._epochs = 0
        self._updates = 0
        self._converged = False



        n = len(data)
        e = 0      # счётчик подряд верных ответов
        idx = 0

        for it in range(self.max_iter):
            if idx == 0:
                self._epochs += 1

            x, sign = data[idx]
            # Скалярное произведение весов на признаки + смещение.
            d = sum(wi * xi for wi, xi in zip(self.w[:-1], x)) + self.w[-1]

            correct = (sign > 0 and d > 0) or (sign < 0 and d < 0)

            if correct:
                e += 1
                if e == n:
                    self._converged = True
                    break
            else:
                # Коррекция весов: w_j += y * x_j, w0 += y.
                for j in range(self.dim):
                    self.w[j] += sign * x[j]
                self.w[-1] += sign

                self.history.append((x, sign, self.w.copy()))
                self._updates += 1
                e = 0

            idx = (idx + 1) % n

        self._trained = True
        return self

    def decision_value(self, *x) -> float:
        """Значение решающей функции D12(x)."""
        if not self._trained:
            raise RuntimeError("Модель не обучена")
        return sum(wi * xi for wi, xi in zip(self.w[:-1], x)) + self.w[-1]

    def predict(self, *x) -> int:
        """Класс 1, если D12 > 0, иначе класс 2."""
        return self.classes[0] if self.decision_value(*x) > 0 else self.classes[1]

    def decision_equation(self) -> str:
        terms = " ".join(f"{wi:+.3f}·X{j+1}" for j, wi in enumerate(self.w[:-1]))
        return f"D12 = {terms} {self.w[-1]:+.3f}"

    def boundary_points(self, x1_min=0.0, x1_max=1.0):
        if self.dim != 2:
            raise RuntimeError(
                "boundary_points работает только при d = 2. "
                f"Текущая размерность: {self.dim}"
            )
        w1, w2, w0 = self.w[0], self.w[1], self.w[-1]
        if abs(w2) > 1e-9:
            x2a = -(w1 * x1_min + w0) / w2
            x2b = -(w1 * x1_max + w0) / w2
            return x1_min, x2a, x1_max, x2b
        elif abs(w1) > 1e-9:
            x1 = -w0 / w1
            return x1, 0.0, x1, 1.0
        else:
            return 0.0, 0.0, 1.0, 0.0