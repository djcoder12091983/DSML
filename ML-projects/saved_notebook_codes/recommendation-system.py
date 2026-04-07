# first we will build some recommendation system based on some synthetic dataset then we will explore the same on real time data

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- Generate High-Volume Synthetic Data ---
def generate_synthetic_data(num_users=1000, num_ingredients=500, density=0.05):
    # Calculate how many 'pairings' to create
    num_interactions = int(num_users * num_ingredients * density)
    
    # Randomly assign users to ingredients
    rows = np.random.randint(0, num_users, num_interactions)
    cols = np.random.randint(0, num_ingredients, num_interactions)
    
    # 'data' represents the frequency or rating (1-10) of using that ingredient
    data = np.random.randint(1, 11, num_interactions)
    
    # Store as a Sparse Matrix (extremely memory efficient for large data)
    return csr_matrix((data, (rows, cols)), shape=(num_users, num_ingredients))

# Create our universe of ingredients
ingredient_list = [f"ingredient_{i}" for i in range(500)]
user_item_matrix = generate_synthetic_data()

# --- The Logic (K-Nearest Neighbors) ---
# We use Cosine Similarity to find ingredients that 'behave' similarly in recipes
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(user_item_matrix.T) # Transpose so we compare ingredients, not users

# --- The Recommendation Function ---
def recommend_pairing(ingredient_name, n_recs=3):
    try:
        idx = ingredient_list.index(ingredient_name)
        # Find the 'closest' neighbors in the vector space
        distances, indices = model_knn.kneighbors(user_item_matrix.T[idx], n_neighbors=n_recs+1)
        
        print(f"🍳 Since you're using {ingredient_name}, try adding:")
        for i in range(1, len(distances.flatten())):
            print(f" -> {ingredient_list[indices.flatten()[i]]} (Match Score: {1 - distances.flatten()[i]:.2f})")
    except ValueError:
        print("Ingredient not found in database!")

# --- Real-Time Test ---
recommend_pairing("ingredient_42")



# TODO explore with real data