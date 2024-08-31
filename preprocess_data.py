import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and prepare data
def preprocess_and_save_data():
    # Load raw data
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    u_data = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', names=columns)
    u_data = u_data.drop('timestamp', axis=1)

    
    columns_item = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 
                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                    'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    u_item = pd.read_csv('ml-100k/ml-100k/u.item', sep='|', names=columns_item, encoding='latin-1')
    
    columns_user = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    u_user = pd.read_csv('./ml-100k/ml-100k/u.user', sep='|', names=columns_user)
    
    user_movie_ratings = u_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    u_user['gender_encoded'] = le_gender.fit_transform(u_user['gender'])
    u_user['occupation_encoded'] = le_occupation.fit_transform(u_user['occupation'])
    
    # Save processed data to files
    with open('data/user_movie_ratings.pkl', 'wb') as f:
        pickle.dump(user_movie_ratings, f)
    
    with open('data/u_user_encoded.pkl', 'wb') as f:
        pickle.dump(u_user, f)
    
    with open('data/u_item.pkl', 'wb') as f:
        pickle.dump(u_item, f)

    with open('data/u_data.pkl', 'wb') as f:
        pickle.dump(u_data, f)
    
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    preprocess_and_save_data()
