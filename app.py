import pickle
import pandas as pd
import streamlit as st
import requests
import numpy as np

# Function to fetch movie poster from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        # Return placeholder if no poster
        return "https://via.placeholder.com/500x750?text=No+Image"

# Recommendation function
def recommend(movie):
    # Case-insensitive search
    filtered = movies[movies['title'].str.lower() == movie.lower()]
    if filtered.empty:
        st.error("Movie not found in database!")
        return [], []

    index = filtered.index[0]

    # Similarity scores from NumPy array
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movie_names = []
    recommended_movie_posters = []

    # Get top 5 recommendations (skip the movie itself)
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters

# Streamlit App
st.header("🎬 Movie Recommender System")

# Load pickle files
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)  # Convert dict to DataFrame

similarity = pickle.load(open("similarity.pkl", "rb"))  # NumPy array

# Movie dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# Show recommendations
if st.button("Show Recommendation"):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    if recommended_movie_names and recommended_movie_posters:
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(recommended_movie_names):
                col.text(recommended_movie_names[idx])
                col.image(recommended_movie_posters[idx])
