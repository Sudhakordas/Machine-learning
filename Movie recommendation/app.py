import streamlit as st
import pandas as pd
import pickle
import requests

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movie = pd.DataFrame(movies_dict)

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=cd92ddabb2407347a74934e7a5fae95c&language=en-US'.format(movie_id))
        
        #requests.get('https://api.themoviedb.org/3/movie/{}api_key=cd92ddabb2407347a74934e7a5fae95c&language=en-US'.format(movie_id))
    
    data = response.json()
    #st.text(data)
    return  "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

def recommend(select_movie):
    movie_index = movie[movie['title'] == select_movie].index[0]
    similar = similarity[movie_index]
    movies_list = sorted(list(enumerate(similar)), reverse=True, key=lambda x: x[1])[1:11]

    recommend_movies = []
    recommended_movies_poster = []
    for i in movies_list:
        movie_id = movie.iloc[i[0]].movie_id
        recommend_movies.append(movie.iloc[i[0]].title)
        #fetch poster
        recommended_movies_poster.append(fetch_poster(movie_id))

    return recommend_movies, recommended_movies_poster

st.title('Movie Recommendation System')

selected_movie = st.selectbox(
    'Select your movie',
    movie['title'].values
)

if st.button('Recommend'):
    
    
    names, posters = recommend(selected_movie)
    
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  = st.columns(10)
    counter = 0
    #for names, posters in recommend(selected_movie):
        
    with col1:
        st.text(names[0])
        st.image(posters[0])
        
    with col2:
        st.text(names[1])
        st.image(posters[1])
        
    with col3:
        st.text(names[2])
        st.image(posters[2])
       
    with col4:
        st.text(names[3])
        st.image(posters[3])
       
    with col5:
        st.text(names[4])
        st.image(posters[4])
       
    with col6:
        st.text(names[5])
        st.image(posters[5])
       
    with col7:
        st.text(names[6])
        st.image(posters[6])
       
    with col8:
        st.text(names[7])
        st.image(posters[7])
       
    with col9:
        st.text(names[8])
        st.image(posters[8])
       
    with col10:
        st.text(names[9])
        st.image(posters[9])
       