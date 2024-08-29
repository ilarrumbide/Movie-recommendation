from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn
import time
from model import RecommendationSystem

recommender = RecommendationSystem()

app = FastAPI()

class User(BaseModel):
    user_id: int
    age: int
    gender: str
    occupation: str
    zip_code: str

@app.post("/add_user")
def api_add_user(user: User):
    if recommender.user_exists(user.user_id):
        raise HTTPException(status_code=400, detail="User ID already exists")
    try:
        recommender.add_new_user(user.user_id, user.age, user.gender, user.occupation, user.zip_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "User added successfully"}

@app.get("/recommend/{user_id}")
def api_recommend(user_id: int):
    start_time = time.time()  # Comienza el temporizador
    if not recommender.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    recommendations = recommender.get_top_n_recommendations(user_id)
    end_time = time.time()  # Finaliza el temporizador
    print(f"Tiempo para predecir ratings para 10 películas: {end_time - start_time:} segundos.")
    return {"user_id": user_id, "recommendations": recommendations}


@app.get("/information/{user_id}")
def info_user(user_id: int):
    if not recommender.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    movies = recommender.get_user_rated_movies(user_id)
    return {"user_id": user_id, "movies": movies}
