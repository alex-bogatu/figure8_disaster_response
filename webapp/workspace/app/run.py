import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    sorted_cat = df.iloc[:, 4:].sum(axis=0).sort_values()
    top_10_values = sorted_cat[-10:][::-1].tolist()
    top_10_cat_names = sorted_cat[-10:][::-1].index.tolist()
    
    type_distribution_values = sorted_cat.tolist()
    type_distribution_names = sorted_cat.index.tolist()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_10_values,
                    y=top_10_cat_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Top 10 Category Types',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        
        {
            'data': [
                Pie(
                    values=type_distribution_values,
                    labels=type_distribution_names,
                )
            ],

            'layout': {
                'title': 'Message types'
            }
        }
    ]    
 
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
