from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import pickle
from surprise.model_selection import GridSearchCV
import os
from surprise import accuracy


# Load preprocessed data
with open('data/user_movie_ratings.pkl', 'rb') as f:
    user_movie_ratings = pickle.load(f)

with open('data/u_user_encoded.pkl', 'rb') as f:
    u_user = pickle.load(f)

with open('data/u_item.pkl', 'rb') as f:
    u_item = pickle.load(f)

with open('data/u_data.pkl', 'rb') as f:
    u_data = pickle.load(f)



# Centrar las calificaciones restando la media del usuario
u_data['rating_centered'] = u_data.groupby('user_id')['rating'].transform(lambda x: x - x.mean())

# Definir el Reader para Surprise
reader = Reader(rating_scale=(u_data['rating_centered'].min(), u_data['rating_centered'].max()))

# Crear el dataset para Surprise usando ratings centrados
data = Dataset.load_from_df(u_data[['user_id', 'item_id', 'rating_centered']], reader)

# Definir la grid de hiperparámetros
param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs': [20, 50],
    'lr_all': [0.005, 0.010],
    'reg_all': [0.02, 0.1]
}

# Búsqueda de hiperparámetros con validación cruzada
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Mejor modelo
best_algo = gs.best_estimator['rmse']

# Imprimir los mejores parámetros
best_params = gs.best_params['rmse']
print(f"Mejores parámetros: {best_params}")

# Validación cruzada con el mejor modelo
results = cross_validate(best_algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Guardar el mejor modelo entrenado
model_filename = 'models/best_svd_model_center.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(best_algo, model_file)

# Guardar las métricas en un archivo .txt
metrics_filename = 'cross_validation_metrics.txt'
with open(metrics_filename, 'w') as metrics_file:
    metrics_file.write(f"Mejores parámetros: {best_params}\n")
    metrics_file.write(f"Resultados de validación cruzada:\n")
    for key, values in results.items():
        metrics_file.write(f"{key}: {values}\n")
