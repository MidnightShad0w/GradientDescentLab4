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

categorical = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage']
numeric = ['powerPS', 'kilometer', 'autoAgeMonths']
other = []

# Обработка пропусков, если необходимо
data = data.dropna(subset=categorical + numeric + ['price'])

# Добавление смещения (bias)
data['bias'] = 1
other += ['bias']

x = data[categorical + numeric + other]
y = data['price']

# Преобразование признаков
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('scaling', StandardScaler(), numeric),
    ('other',  'passthrough', other)
])

x = column_transformer.fit_transform(x)

# Разделение данных
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/9, random_state=42)

print(f"Размер обучающей выборки: {x_train.shape[0]}")
print(f"Размер валидационной выборки: {x_val.shape[0]}")
print(f"Размер тестовой выборки: {x_test.shape[0]}")

# Определение параметров для подбора
lambdas = np.logspace(-5, 1, 5)  # Длина шага
mus = np.logspace(-5, 1, 5)      # Коэффициент регуляризации
methods = ['full', 'momentum', 'adam']
regularization_options = [False, True]

tolerance = 1e-4
max_iter = 500

# Инициализация структуры для хранения результатов
results = {}

# Цикл по методам градиентного спуска
for method in methods:
    results[method] = {}
    for regularized in regularization_options:
        reg_label = 'reg' if regularized else 'no_reg'
        results[method][reg_label] = {}
        if regularized:
            parameter_grid = itertools.product(lambdas, mus)
        else:
            parameter_grid = itertools.product(lambdas, [0])  # mu=0 для отсутствия регуляризации

        for lambda_, mu in parameter_grid:
            # Конфигурация градиентного спуска
            descent_kwargs = {
                'dimension': x_train.shape[1],
                'lambda_': lambda_,
            }
            if regularized:
                descent_kwargs['mu'] = mu

            descent_config = {
                'descent_name': method,
                'regularized': regularized,
                'kwargs': descent_kwargs
            }

            # Обучение модели
            model = LinearRegression(descent_config=descent_config, tolerance=tolerance, max_iter=max_iter)
            model.fit(x_train, y_train)

            # Прогнозирование
            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)

            # Вычисление метрик
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            n_iterations = len(model.loss_history) - 1

            # Сохранение результатов
            param_key = f"lambda_{lambda_:.1e}_mu_{mu:.1e}"
            results[method][reg_label][param_key] = {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'n_iterations': n_iterations,
                'loss_history': model.loss_history
            }

            print(f"Метод: {method}, Регуляризация: {reg_label}, λ: {lambda_:.1e}, μ: {mu:.1e}, "
                  f"Итераций: {n_iterations}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, "
                  f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

# Визуализация результатов
for method in methods:
    for reg_label in ['no_reg', 'reg']:
        plt.figure(figsize=(12, 8))
        for param_key, metrics in results[method][reg_label].items():
            history = metrics['loss_history']
            label = param_key
            plt.plot(range(len(history)), history, label=label)
        title_reg = "Без Регуляризации" if reg_label == 'no_reg' else "С Регуляризацией"
        plt.title(f"Метод: {method} — Зависимость функции потерь (MSE) от итераций ({title_reg})")
        plt.xlabel("Итерация")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()

# Анализ лучших параметров
for method in methods:
    print(f"\nЛучшие параметры для метода: {method}")
    for reg_label in ['no_reg', 'reg']:
        if reg_label == 'no_reg':
            # Находим модель с минимальным val_mse без регуляризации
            best_param = min(results[method][reg_label].items(), key=lambda x: x[1]['val_mse'])[0]
        else:
            # Находим модель с минимальным val_mse среди регуляризованных
            best_param = min(results[method][reg_label].items(), key=lambda x: x[1]['val_mse'])[0]
        best_metrics = results[method][reg_label][best_param]
        print(f"  Регуляризация: {reg_label}")
        print(f"    Параметры: {best_param}")
        print(f"    Val MSE: {best_metrics['val_mse']:.4f}")
        print(f"    Val R2: {best_metrics['val_r2']:.4f}")
        print(f"    Итераций: {best_metrics['n_iterations']}")

# Дополнительно: Тестирование на Тестовой Выборке
for method in methods:
    for reg_label in ['no_reg', 'reg']:
        # Найти лучшую модель
        if reg_label == 'no_reg':
            best_param = min(results[method][reg_label].items(), key=lambda x: x[1]['val_mse'])[0]
            mu = 0
        else:
            best_param = min(results[method][reg_label].items(), key=lambda x: x[1]['val_mse'])[0]
            mu = float(best_param.split('_mu_')[1])

        lambda_ = float(best_param.split('_mu_')[0].split('_')[-1])

        # Конфигурация градиентного спуска
        descent_kwargs = {
            'dimension': x_train.shape[1],
            'lambda_': lambda_,
        }
        if reg_label == 'reg':
            descent_kwargs['mu'] = mu

        descent_config = {
            'descent_name': method,
            'regularized': reg_label == 'reg',
            'kwargs': descent_kwargs
        }

        # Обучение модели на обучающей + валидационной выборках
        combined_x = np.vstack((x_train, x_val))
        combined_y = np.concatenate((y_train, y_val))

        model = LinearRegression(descent_config=descent_config, tolerance=tolerance, max_iter=max_iter)
        model.fit(combined_x, combined_y)

        # Прогнозирование на тестовой выборке
        y_test_pred = model.predict(x_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\nМетод: {method}, Регуляризация: {reg_label}, Параметры: {best_param}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test R2: {test_r2:.4f}")
