from typing import List, Tuple, Dict,Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from functools import lru_cache

class RecommendationSystem:
    def __init__(self):
        self.u_user = None
        self.u_item = None
        self.u_data = None
        self.user_movie_ratings = None
        self.loaded_model = None
        self.le_gender = LabelEncoder()
        self.le_occupation = LabelEncoder()

    @staticmethod
    def load_pickle_file(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def load_data(self):
        if self.u_user is None:
            self.u_user = self.load_pickle_file('data/u_user_encoded.pkl')
            self.u_item = self.load_pickle_file('data/u_item.pkl')
            self.u_data = self.load_pickle_file('data/u_data.pkl')
            self.user_movie_ratings = self.load_pickle_file('data/user_movie_ratings.pkl')
            self.loaded_model = self.load_pickle_file('models/best_svd_model_center.pkl')
            
            self.u_user['gender_encoded'] = self.le_gender.fit_transform(self.u_user['gender'])
            self.u_user['occupation_encoded'] = self.le_occupation.fit_transform(self.u_user['occupation'])

    def add_new_user(self, user_id: int, age: int, gender: str, occupation: str, zip_code: str):
        self.load_data()
        if gender not in self.le_gender.classes_:
            raise ValueError(f"Invalid gender '{gender}'. Valid values: {', '.join(self.le_gender.classes_)}.")
        if occupation not in self.le_occupation.classes_:
            raise ValueError(f"Invalid occupation '{occupation}'. Valid values: {', '.join(self.le_occupation.classes_)}.")
        
        new_user = pd.DataFrame({
            'user_id': [user_id],
            'age': [age],
            'gender': [gender],
            'occupation': [occupation],
            'zip_code': [zip_code],
            'gender_encoded': self.le_gender.transform([gender]),
            'occupation_encoded': self.le_occupation.transform([occupation])
        })
        
        self.u_user = pd.concat([self.u_user, new_user], ignore_index=True)
        print("New user added successfully.")

    @lru_cache(maxsize=128)
    def get_top_n_recommendations(self, user_id: int, n: int = 10) -> List[Tuple[str, float]]:
        self.load_data()
        if user_id not in self.user_movie_ratings.index:
            print("Using cold start")
            return self.get_cold_start_recommendations(user_id, n)
        
        user_rated_items = set(self.u_data[self.u_data['user_id'] == user_id]['item_id'])
        items_not_rated = self.u_item[~self.u_item['movie_id'].isin(user_rated_items)]
        
        if items_not_rated.empty:
            print(f'No unrated items for user ID {user_id}.')
            return []

        user_mean = self.u_data[self.u_data['user_id'] == user_id]['rating'].mean()

        movie_ids = items_not_rated['movie_id'].values
        predicted_ratings = np.array([self.loaded_model.predict(user_id, movie_id).est for movie_id in movie_ids])
        predicted_ratings += user_mean

        top_n_indices = np.argsort(predicted_ratings)[-n:][::-1]
        top_n_movies = movie_ids[top_n_indices]
        top_n_ratings = predicted_ratings[top_n_indices]

        result = [(self.u_item.loc[self.u_item['movie_id'] == movie_id, 'movie_title'].iloc[0], rating) 
                  for movie_id, rating in zip(top_n_movies, top_n_ratings)]
        return result

    def get_cold_start_recommendations(self, user_id: int, n: int = 10) -> List[Tuple[str, float]]:
        self.load_data()
        user_features = self.u_user[['age', 'gender_encoded', 'occupation_encoded']]
        scaler = StandardScaler()
        user_features_normalized = scaler.fit_transform(user_features)
        cosine_sim = cosine_similarity(user_features_normalized)

        user_index = self.u_user[self.u_user['user_id'] == user_id].index[0]
        user_similarities = cosine_sim[user_index]
        similar_users_indices = user_similarities.argsort()[::-1][1:51]
        similar_users = self.u_user.iloc[similar_users_indices]['user_id'].tolist()
        
        if not similar_users:
            popular_movies = self.u_data.groupby('item_id')['rating'].mean().sort_values(ascending=False)
            recommended_movies = [(self.u_item[self.u_item['movie_id'] == i]['movie_title'].iloc[0], r) 
                                  for i, r in popular_movies.head(n).items()]
        else:
            similar_ratings = self.u_data[self.u_data['user_id'].isin(similar_users)]
            movie_ratings = similar_ratings.groupby('item_id')['rating'].mean().sort_values(ascending=False)
            recommended_movies = [(self.u_item[self.u_item['movie_id'] == i]['movie_title'].iloc[0], r) 
                                  for i, r in movie_ratings.head(n).items()]
        
        return recommended_movies

    def user_exists(self, user_id: int) -> bool:
        self.load_data()
        return user_id in self.u_user['user_id'].values

    def get_user_rated_movies(self, user_id: int) -> List[Dict[str, Union[str, float]]]:
        self.load_data()
        user_ratings = self.u_data[self.u_data['user_id'] == user_id]
        if user_ratings.empty:
            return []

        user_movies = user_ratings.merge(self.u_item[['movie_id', 'movie_title']], left_on='item_id', right_on='movie_id')
        user_movies = user_movies.sort_values('rating', ascending=False)
        return user_movies[['movie_title', 'rating']].to_dict(orient='records')






























# def get_top_n_recommendations(user_id, n=10):
#     # Definir nombres de columna correctos
#     item_column = 'item_id'  # En u_data
#     movie_column = 'movie_id'  # En u_item
#     movie_title_column = 'movie_title'

#     algo = loaded_model
    
#     # Verificar si el usuario está en u_data
#     if user_id not in user_movie_ratings.index:
#         print("Utilizando cold start")
#         return get_cold_start_recommendations(user_id, n)
    
#     # Obtener todos los ítems que el usuario no ha calificado
#     items_not_rated = u_item[~u_item[movie_column].isin(u_data[u_data['user_id'] == user_id][item_column])]
    
#     # Verificar si hay ítems no calificados
#     if items_not_rated.empty:
#         print(f'No hay ítems no calificados para el usuario ID {user_id}.')
#         return []
    
#     # Obtener la media del usuario (para deshacer el centrado)
#     user_mean = u_data[u_data['user_id'] == user_id]['rating'].mean()

#     # Predecir el rating para todos esos ítems
#     predictions = []
#     for movie_id in items_not_rated[movie_column]:
#         try:
#             # Predecir el rating centrado
#             predicted_rating_centered = algo.predict(user_id, movie_id).est
#             # Deshacer el centrado sumando la media del usuario
#             predicted_rating = predicted_rating_centered + user_mean
#         except Exception as e:
#             print(f'Error al predecir el rating para usuario {user_id} y película {movie_id}: {e}')
#             continue
#         predictions.append((movie_id, predicted_rating))
    
#     # Ordenar por los ratings más altos
#     top_n_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
#     # Unir con u_item para obtener los títulos de las películas
#     top_n_items_with_titles = []
#     for movie_id, predicted_rating in top_n_items:
#         movie_title = u_item[u_item[movie_column] == movie_id][movie_title_column].values[0]
#         top_n_items_with_titles.append((movie_title, predicted_rating))
    
#     return top_n_items_with_titles