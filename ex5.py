# Тут просто изучал всякое

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from descents import get_descent
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('autos.csv')

categorical = ['brand', 'model', 'vehicleType', 'gearbox', 'fuelType', 'notRepairedDamage']
numeric = ['powerPS', 'kilometer', 'autoAgeMonths']
other = []

data.isnull().sum()

data['bias'] = 1
other += ['bias']

x = data[categorical + numeric + other]
y = data['price']

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('scaling', StandardScaler(), numeric),
    ('other',  'passthrough', other)
])

x = column_transformer.fit_transform(x)

x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/9, random_state=42)

print(f"Размер обучающей выборки: {x_train.shape[0]}")
print(f"Размер валидационной выборки: {x_val.shape[0]}")
print(f"Размер тестовой выборки: {x_test.shape[0]}")


lambdas = np.logspace(-5, -1, 10)  # тут при лямбда ~= 1.0e+01 loss улетают в небеса
results = {}
methods = ['full', 'momentum', 'adam']

tolerance = 1e-4  # тут решил поиграться с tolerance и max_iter
max_iter = 500

for method in methods:
    results[method] = {}
    for lambda_ in lambdas:
        descent_config = {
            'descent_name': method,
            'regularized': False,
            'kwargs': {
                'dimension': x_train.shape[1],
                'lambda_': lambda_
            }
        }

        model = LinearRegression(descent_config=descent_config, tolerance=tolerance, max_iter=max_iter)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        n_iterations = len(model.loss_history) - 1

        results[method][lambda_] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'n_iterations': n_iterations,
            'loss_history': model.loss_history
        }

        print(f"Метод: {method}, λ: {lambda_:.1e}, Итераций: {n_iterations}, "
              f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, "
              f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

for method in methods:
    plt.figure(figsize=(12, 8))
    for lambda_ in lambdas:
        history = results[method][lambda_]['loss_history']
        plt.plot(range(len(history)), history, label=f"λ={lambda_:.1e}")
    plt.title(f"Метод: {method} — Зависимость ошибки от количества итераций")
    plt.xlabel("Итерация")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale('log')
    plt.show()
