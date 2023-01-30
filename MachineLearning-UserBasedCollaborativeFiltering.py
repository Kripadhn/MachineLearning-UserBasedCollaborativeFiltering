import pandas as pd
from surprise import Reader, Dataset, KNNWithMeans

# Load the data
ratings_data = pd.read_csv("ratings.csv")

# Prepare the data for the algorithm
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)

# Train the model
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict the rating for a user and item
prediction = algo.predict(1, 3)
print("Predicted rating:", prediction[3])
