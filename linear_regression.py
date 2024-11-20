from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # TODO: реализовать подбор весов для x и y
        loss = self.calc_loss(x, y)  # Вычисляем начальное значение функции потерь
        self.loss_history.append(loss)  # Добавляем в историю потерь

        for iteration in range(self.max_iter):

            # Выполняем шаг градиентного спуска
            delta_w = self.descent.step(x, y)  # delta_w = w[k+1] - w[k]

            # Проверяем на NaN в весах
            if np.isnan(self.descent.w).any():
                print(f"№ итерации - {iteration}, в весах появились NaN.")
                break

            if np.isnan(loss) or np.isinf(loss):
                print(f"На итерации {iteration} обнаружено некорректное значение функции потерь.")
                break

            # Вычисляем евклидову норму разницы весов
            weight_diff_norm = np.linalg.norm(delta_w)

            # Записываем текущее значение функции потерь в историю
            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

            # Проверяем критерий остановки по tolerance
            if weight_diff_norm < self.tolerance:
                print(f"Критерий сходимости достигнут на итерации {iteration}.")
                break

        else:
            # Если цикл завершился без break, то достигнут max_iter
            print(f"Достигнуто максимальное количество итераций: {self.max_iter}")

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
