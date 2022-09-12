from flask import Flask, render_template, request
import pickle 
import numpy as np
import pandas as pd


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')



@app.route('/Predict', methods = ['POST'])
def home():
    
    #input_features = [float(x) for x in request.form.values()]
    #features_values = [np.array(input_features)]
    #print(features_values)
    
    data = request.form['a'] 
    
    def movie_recommendations(movie_user_likes):
    
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        df = pd.read_csv('IMDB-Movie-Data.csv')

        #feature selection
        feature = ['Title','Genre','Director','Actors']

         #Creating a column which can combine all the selected features.
        def combine_features(row):
            return row['Title'] + " "+row['Genre'] +" "+ row['Director'] +" "+ row['Actors']
        #applying this method into our dataset
        df['Combine'] = df.apply(combine_features,axis = 1)

        #creating the count matrix from new combine features.
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df['Combine'])

        #getting the cosine similarity in the count matrix.
        cosine_sim =  cosine_similarity(count_matrix)

        #to give similar movies we need the index of the movie.We can have bu seeing the rank features.
        def get_index_from_title(title):
            return df[df.Title == title]['Rank'].values[0]

        #getting the index of that movie
        movie_index = get_index_from_title(movie_user_likes)

        #enumerate the list 
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        #sorting the list of tupples in descending order.
        sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1], reverse = True)

        #function for getting the title of the similar movie from index
        def get_title_from_index(index):
            return df[df.Rank == index]['Title'].values[0]

        #loop for printing similar movie
        i = 0
        for movie in sorted_similar_movies:
            print(get_title_from_index(movie[0]))
            i = i+1
            if i>10:
                break
    
    #data2 = request.form['b']
    #data3 = request.form['c']
    #data4 = request.form['d']
    
    
    pred = movie_recommendations('data')
    
    
    return render_template('Predict.html',data = pred)
    
    

if __name__ == "__main__":
    app.run(debug = True)