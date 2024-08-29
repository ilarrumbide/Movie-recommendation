from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy
import pickle
import pandas as pd
import numpy as np

# Cargar el dataset MovieLens 1M
data = Dataset.load_builtin('ml-1m')

# Convertir el dataset en DataFrame para aplicar el preprocesamiento
df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'item_id', 'rating','tiempo'])
df['rating'] = df['rating'].astype(float)

# Centrar las calificaciones restando la media del usuario
df['rating_centered'] = df.groupby('user_id')['rating'].transform(lambda x: x - x.mean())

# Definir el Reader para Surprise
reader = Reader(rating_scale=(df['rating_centered'].min(), df['rating_centered'].max()))

# Crear el dataset para Surprise usando ratings centrados
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating_centered']], reader)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
trainset, testset = train_test_split(data, test_size=0.25)

# Cargar el mejor modelo entrenado desde el archivo
with open('models/best_svd_model_center.pkl', 'rb') as model_file:
    best_algo = pickle.load(model_file)

# Realizar las predicciones en el conjunto de prueba
predictions = best_algo.test(testset)

# Evaluar el rendimiento del modelo en el espacio centrado
rmse_centered = accuracy.rmse(predictions)
mae_centered = accuracy.mae(predictions)

print(f'RMSE en datos centrados: {rmse_centered}')
print(f'MAE en datos centrados: {mae_centered}')