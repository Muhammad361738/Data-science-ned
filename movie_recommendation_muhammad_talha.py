
"""# New Section

## IMPORTS
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

"""## CREATING DATAFRAME"""

initail_df = pd.read_csv('movies_recommendation.csv')
initail_df.head(2)

"""## DATASET INFO"""

initail_df.shape

initail_df.describe()

initail_df.info()

"""**NULL VALUES**"""

initail_df.isnull().sum()

"""# VISUALIZATIONS"""

visulaization_df = initail_df.copy()
# visulaization_df.dropna(inplace=True)

def millions(x, pos):
    return '$%1.1fM' % (x * 1e-6)

formatter = FuncFormatter(millions)

"""### TOP 10 HIGHEST BUGHETS MOVIES"""

bughet_sorted_movies = visulaization_df.sort_values(by='Movie_Budget', ascending=False)
top_10_movies = bughet_sorted_movies.iloc[:10]

plt.figure(figsize=(10,6))
ax = sns.barplot(x = 'Movie_Title',
            y = 'Movie_Budget',
            data = top_10_movies)


ax.yaxis.set_major_formatter(formatter)
plt.xticks(rotation=90)
plt.title('Highest Bughet Movies')
plt.show()

"""### Top 10 Movies by Revenue"""

top_10_movies_revenue = visulaization_df.sort_values(by='Movie_Revenue', ascending=False).iloc[:10]

plt.figure(figsize=(10,6))
ax = sns.barplot(x = 'Movie_Title',
            y = 'Movie_Revenue',
            data = top_10_movies_revenue)

plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(formatter)
plt.title('Top 10 Blockbuster Movies')
plt.show()

"""### TOP 10 RATING MOVIES"""

top_10_movies_rating = visulaization_df.sort_values(by='Movie_Vote', ascending=False)[:10]
top_10_movies_rating
plt.figure(figsize=(10,6))
ax = sns.barplot(x = 'Movie_Vote',
            y = 'Movie_Title',
            data = top_10_movies_rating, palette='viridis')

plt.xlabel('Movie Rating')
plt.title('Top 10 Rating Movies')
plt.show()

"""### Top Language in Movies"""

movie_language_count = visulaization_df.groupby('Movie_Language').size().reset_index(name='Count')
movie_language_count = movie_language_count.sort_values(by='Count', ascending=False)

top_1 = movie_language_count.head(1)
others = movie_language_count.iloc[5:]

others_sum = others['Count'].sum()

other_row = pd.DataFrame({'Movie_Language': ['Others'], 'Count': [others_sum] })
movie_language_count = pd.concat([top_1, other_row], ignore_index=True)

plt.pie(movie_language_count['Count'], labels=movie_language_count['Movie_Language'],
        autopct='%.0f%%')
plt.title('Top Language in Movies')
plt.show()

"""# TRANSFORMING DATA

### DROPPING UNNECESSARY COLUMNS
"""

feature_cols = ['Movie_Title', 'Movie_Genre', 'Movie_Keywords', 'Movie_Overview', 'Movie_Tagline',
                'Movie_Cast', 'Movie_Director']

df = initail_df[feature_cols]

"""### HANDLING NULL VALUES"""

df = df[feature_cols].fillna('')

"""### Combine all selected features into one"""

df['combined_features'] = df['Movie_Genre'] + ' ' + df['Movie_Keywords'] + ' ' + \
                          df['Movie_Overview'] + ' ' + df['Movie_Tagline'] + ' ' + \
                          df['Movie_Cast'] + ' ' + df['Movie_Director']

"""# Vectorization"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
# print(tfidf_vectorizer.vocabulary_)
# print(tfidf_matrix)

"""# KNN MODEL TRAINING"""

from sklearn.neighbors import NearestNeighbors

knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

"""# MOVIES SUGGESTIONS"""

def get_movie_index(title):
    return df[df['Movie_Title'] == title].index[0]

def get_similar_movies(movie_title):
    movie_idx = get_movie_index(movie_title)

    distances, indices = knn_model.kneighbors(tfidf_matrix[movie_idx], n_neighbors=10)

    similar_movies = []
    for i in range(1, len(indices[0])):
        similar_movies.append(df['Movie_Title'].iloc[indices[0][i]])

    return similar_movies

input_movie = input('Enter Movie Name')

similar_movies = get_similar_movies(input_movie)
pd.DataFrame({'Similar Movies': similar_movies})

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommender | Muhammad Talha", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>by <strong>Muhammad Talha</strong></p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Dataset ---
try:
    initail_df = pd.read_csv('movies_recommendation.csv')
    st.success("✅ Dataset loaded successfully.")
except FileNotFoundError:
    st.error("❌ File 'movies_recommendation.csv' not found.")
    st.stop()

# --- Tabs for Clean Navigation ---
tab1, tab2, tab3 = st.tabs(["🔍 Recommend Movies", "🎭 Browse by Genre", "📊 Dataset Overview"])

with tab1:
    st.subheader("🎥 Find Similar Movies")
    feature_cols = ['Movie_Title', 'Movie_Genre', 'Movie_Keywords', 'Movie_Overview',
                    'Movie_Tagline', 'Movie_Cast', 'Movie_Director']
    df = initail_df[feature_cols].fillna('')

    df['combined_features'] = (
        df['Movie_Genre'] + ' ' +
        df['Movie_Keywords'] + ' ' +
        df['Movie_Overview'] + ' ' +
        df['Movie_Tagline'] + ' ' +
        df['Movie_Cast'] + ' ' +
        df['Movie_Director']
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

    knn_model = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn_model.fit(tfidf_matrix)

    selected_movie = st.selectbox("🎬 Choose a movie:", df['Movie_Title'].tolist())

    if selected_movie:
        movie_idx = df[df['Movie_Title'] == selected_movie].index[0]
        distances, indices = knn_model.kneighbors(tfidf_matrix[movie_idx], n_neighbors=11)
        similar_movies = df['Movie_Title'].iloc[indices[0][1:]].tolist()

        st.success(f"🎉 Recommendations based on: **{selected_movie}**")
        st.table(pd.DataFrame({'Recommended Movies': similar_movies}))

with tab2:
    st.subheader("🎭 Browse by Genre")
    if 'Movie_Genre' in df.columns:
        all_genres = df['Movie_Genre'].dropna().unique()
        selected_genre = st.selectbox("📌 Select a genre", sorted(all_genres))

        genre_filtered = initail_df[initail_df['Movie_Genre'] == selected_genre]
        st.info(f"🎬 Movies in genre: **{selected_genre}**")
        st.dataframe(genre_filtered[['Movie_Title', 'Movie_Genre']], use_container_width=True)

with tab3:
    st.subheader("📊 Dataset Summary")

    with st.expander("🧾 View Basic Info"):
        st.write(initail_df.info())

    col1, col2 = st.columns(2)

    with col1:
        st.write("🔢 Dataset Shape")
        st.code(f"{initail_df.shape}")

        st.write("📌 Missing Values")
        st.dataframe(initail_df.isnull().sum().to_frame("Missing Count"))

    with col2:
        st.write("📊 Data Description")
        st.dataframe(initail_df.describe())

    with st.expander("🧮 Preview First 10 Rows"):
        st.dataframe(initail_df.head(10))

st.markdown("---")
st.caption("🚀 Built with Streamlit | Maintained by Muhammad Talha")
