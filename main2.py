from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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

# разделяем временную выборку на обучающую и валидационную (8:1)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/9, random_state=42)

print(f"Размер обучающей выборки: {x_train.shape[0]}")
print(f"Размер валидационной выборки: {x_val.shape[0]}")
print(f"Размер тестовой выборки: {x_test.shape[0]}")


y_train = y_train.values
batch_sizes = np.arange(5, 500, 10)

k_runs = 10

avg_times = []
avg_iterations = []

tolerance = 1e-4
max_iter = 500

for batch_size in batch_sizes:
    times = []
    iterations = []
    for run in range(k_runs):
        start_time = time.time()

        descent_config = {
            'descent_name': 'stochastic',
            'regularized': False,
            'kwargs': {
                'dimension': x_train.shape[1],
                'lambda_': 0.01,
                'batch_size': batch_size
            }
        }

        model = LinearRegression(
            descent_config=descent_config,
            tolerance=tolerance,
            max_iter=max_iter
        )

        model.fit(x_train, y_train)

        end_time = time.time()

        times.append(end_time - start_time)
        iterations.append(len(model.loss_history) - 1)

    avg_time = np.mean(times)
    avg_iter = np.mean(iterations)

    avg_times.append(avg_time)
    avg_iterations.append(avg_iter)

    print(f"Размер батча: {batch_size}, Среднее время обучения: {avg_time:.4f} сек, "
          f"Среднее число итераций: {avg_iter:.1f}")


plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, avg_iterations, marker='o')
plt.title("Зависимость среднего количества итераций до сходимости от размера батча")
plt.xlabel("Размер батча")
plt.ylabel("Среднее количество итераций")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, avg_times, marker='o', color='orange')
plt.title('Зависимость среднего времени обучения от размера батча')
plt.xlabel('Размер батча')
plt.ylabel('Среднее время обучения (секунды)')
plt.grid(True)
plt.show()