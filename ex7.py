from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from descents import get_descent
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Загрузка и подготовка данных
data = pd.read_csv('autos.csv')

# Определение категориальных и числовых признаков
categorical = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage']
numeric = ['powerPS', 'kilometer', 'autoAgeMonths']
other = []

# Добавление смещения (bias)
data['bias'] = 1
other += ['bias']

# Формирование матрицы признаков и целевой переменной
x = data[categorical + numeric + other]
y = data['price']

# Преобразование признаков с использованием OneHotEncoder и StandardScaler
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('scaling', StandardScaler(), numeric),
    ('other', 'passthrough', other)
])

# Преобразование и конвертация в плотный массив
x = column_transformer.fit_transform(x)

# Проверка формы признаков после преобразования
print(f"Форма признаков после преобразования: {x.shape}")

# Разделение данных на обучающую, валидационную и тестовую выборки
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1 / 9, random_state=42)

# Проверка форм обучающих и валидационных данных
print(f"Форма обучающей выборки: {x_train.shape}")
print(f"Форма валидационной выборки: {x_val.shape}")
print(f"Форма тестовой выборки: {x_test.shape}")

# Определение параметров для подбора
lambdas = 0.1  # Скорость обучения лучшая, судя по графикам - 0.1
mus = np.logspace(-5, 1, 5)  # Коэффициент регуляризации от 1e-5 до 1e+1
methods = ['full', 'momentum', 'adam']

tolerance = 1e-4
max_iter = 500

# Инициализация структуры для хранения результатов
# Структура: results[method][mu][lambda_] = metrics
results = {method: {mu: {} for mu in mus} for method in methods}

# Цикл по методам градиентного спуска
for method in methods:
    for mu in mus:
        # Конфигурация градиентного спуска с регуляризацией
        descent_kwargs = {
            'dimension': x_train.shape[1],
            'lambda_': lambdas,
            'mu': mu  # Регуляризация всегда включена
        }

        descent_config = {
            'descent_name': method,
            'regularized': True,  # Регуляризация включена
            'kwargs': descent_kwargs
        }

        # Обучение модели
        model = LinearRegression(descent_config=descent_config, tolerance=tolerance, max_iter=max_iter)
        model.fit(x_train, y_train)

        # Прогнозирование на обучающей и валидационной выборках
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)

        # Вычисление метрик
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        n_iterations = len(model.loss_history) - 1

        # Сохранение результатов
        results[method][mu] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'n_iterations': n_iterations,
            'loss_history': model.loss_history
        }

        print(f"Метод: {method}, μ: {mu:.1e}, λ: {lambdas:.1e}, "
              f"Итераций: {n_iterations}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, "
              f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

# Визуализация результатов
for method in methods:
    for mu in mus:
        plt.figure(figsize=(12, 8))
        for lambda_, metrics in results[method][mu].items():
            history = metrics['loss_history']
            label = f"λ={lambda_:.1e}"
            plt.plot(range(len(history)), history, label=label)
        plt.title(f"Метод: {method} — μ={mu:.1e} — Зависимость функции потерь (MSE) от итераций")
        plt.xlabel("Итерация")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()

# Анализ лучших параметров
for method in methods:
    print(f"\nЛучшие параметры для метода: {method}")
    for mu in mus:
        # Находим модель с минимальным val_mse для текущего mu
        best_lambda = min(results[method][mu].items(), key=lambda x: x[1]['val_mse'])[0]
        best_metrics = results[method][mu][best_lambda]
        print(f"  μ: {mu:.1e}")
        print(f"    λ: {best_lambda:.1e}")
        print(f"    Val MSE: {best_metrics['val_mse']:.4f}")
        print(f"    Val R2: {best_metrics['val_r2']:.4f}")
        print(f"    Итераций: {best_metrics['n_iterations']}")

# Дополнительно: Тестирование на Тестовой Выборке
for method in methods:
    for mu in mus:
        # Найти лучшую модель для текущего mu
        best_lambda = min(results[method][mu].items(), key=lambda x: x[1]['val_mse'])[0]

        # Конфигурация градиентного спуска с лучшими параметрами
        descent_kwargs = {
            'dimension': x_train.shape[1],
            'lambda_': best_lambda,
            'mu': mu
        }

        descent_config = {
            'descent_name': method,
            'regularized': True,
            'kwargs': descent_kwargs
        }

        # Обучение модели на объединённых обучающей и валидационной выборках
        combined_x = np.vstack((x_train, x_val))
        combined_y = np.concatenate((y_train, y_val))

        model = LinearRegression(descent_config=descent_config, tolerance=tolerance, max_iter=max_iter)
        model.fit(combined_x, combined_y)

        # Прогнозирование на тестовой выборке
        y_test_pred = model.predict(x_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\nМетод: {method}, μ: {mu:.1e}, λ: {best_lambda:.1e}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test R2: {test_r2:.4f}")
