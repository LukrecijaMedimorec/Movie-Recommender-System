from flask import Flask, jsonify, request
import pandas as pd
import heapq
import random
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD


class ContentBasedRecommender:
    def __init__(self, movie_ids, cosine_similarities):
        self.movie_ids = list(set(movie_ids))
        self.cosine_similarities = cosine_similarities

    def get_content_based_recommendations(self, liked_movie_ids, disliked_movie_ids):
        seen_movies = liked_movie_ids + disliked_movie_ids
        liked_movies_indices = [self.movie_ids.index(movie_id) for movie_id in liked_movie_ids]

        all_recommendations = []

        N = 10

        for index in liked_movies_indices:
            similarity_scores = self.cosine_similarities[index]

            valid_top_indices = [i for i in heapq.nlargest(N, range(len(similarity_scores)), key=similarity_scores.__getitem__) if i != index and similarity_scores[i] != 1.0]

            all_recommendations.extend(valid_top_indices)

        random_10_recommendations = random.sample(all_recommendations, N)
        
        recommended_movie_ids = [self.movie_ids[index] for index in random_10_recommendations]

        return recommended_movie_ids



class CollaborativeFilteringRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.reader = Reader(rating_scale=(0, 1))
        self.data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'binary_rating']], self.reader)
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2, random_state=42)
        self.model = None
        self.movie_mapping = None

    def create_movie_mapping(self):
        
        unique_movie_ids = self.ratings_df['movieId'].unique()

        self.movie_mapping = pd.DataFrame({'movieId': unique_movie_ids, 'movie_column': range(len(unique_movie_ids))})


    def recommend_for_user(self, user_id, n=10):
        if self.movie_mapping is None:
            self.create_movie_mapping()

        all_movie_ids = self.ratings_df['movieId'].unique()

        rated_movie_ids = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].tolist()

        unrated_movie_ids = list(set(all_movie_ids) - set(rated_movie_ids))

        predictions = [self.model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]

        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

        top_n_recommendations = [prediction.iid for prediction in sorted_predictions[:n]]
        
        return top_n_recommendations
    
#######Controller
    
app = Flask(__name__)

loaded_model = joblib.load('./models/content_based_recommender.joblib')

movie_ids = loaded_model.movie_ids
cosine_similarities = loaded_model.cosine_similarities

content_based_recommender = ContentBasedRecommender(movie_ids, cosine_similarities)


collabRecommender = joblib.load('./models/collaborative_filtering_model.joblib')

collaborative_filtering_recommender = CollaborativeFilteringRecommender(collabRecommender.ratings_df)
collaborative_filtering_recommender.model = collabRecommender.model 

@app.route('/contentBasedPrediction', methods=['GET'])
def get_content_based_predictions():
    prediction_request = request.json

    if 'liked_movie_ids' not in prediction_request or 'disliked_movie_ids' not in prediction_request:
        return jsonify({"error": "Please provide a valid JSON request"}), 400

    recommendations = content_based_recommender.get_content_based_recommendations(
        prediction_request["liked_movie_ids"], prediction_request["disliked_movie_ids"]
    )

    return jsonify(recommendations)


@app.route('/userBasedPrediction', methods=['GET'])
def get_collaborative_filtering_recommendations():
    prediction_request = request.json

    if 'user_id' not in prediction_request:
        return jsonify({"error": "Please provide a valid JSON request"}), 400
    
    collaborative_recommendations = collaborative_filtering_recommender.recommend_for_user('user_id')

    collaborative_recommendations = [int(x) for x in collaborative_recommendations

]
    return jsonify(collaborative_recommendations)

if __name__=='__main__':
    app.run(port=5000)