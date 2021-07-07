from flask import Flask, request
from flask_restful import Resource, Api
import numpy as np
import pickle
import re
import nltk
# download and import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)
api = Api(app)

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
corpus = pickle.load(open('corpus(1).sav', 'rb'))


class Spam(Resource):
    '''The Resource Spam resource will get messages & shit'''
    def post(self):
        # new_text = 'I hate this restaurant so much'
        new_text = request.get_json()
        print(new_text)
        new_text = new_text['message']
        new_text = re.sub('[^a-zA-Z0-9]', ' ', new_text)
        new_text = new_text.lower()
        new_text = new_text.split()
        ps = PorterStemmer()
        
        all_stopwords = stopwords.words('english')
        exclude = ['not', 'our', "we", "our", "you", "your", "yourself", "it",
               "what", "which", "who", "whom", "this", "that", "these", "those", "until", "while",                         "about", "against", "between", "through", "during", "before", "after", "above", "below",
               "from", "up", "down", "out", "on", "off", "over", "under", "again", "further", "then",
               "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
               "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
               "very", "will", "just", "should", "now"]
        for sw in exclude:
            if sw in all_stopwords:
            # if the words in the exclude list are in the stopword list, then remove them.
                all_stopwords.remove(sw)
        
        new_text = [ps.stem(word) for word in new_text if not word in set(all_stopwords)]
        new_text = ' '.join(new_text)
        new_corpus = [new_text]
        cv = CountVectorizer(max_features=7200)
        X = cv.fit_transform(corpus).toarray()
        new_X_test = cv.transform(new_corpus).toarray()
        new_y_pred = loaded_model.predict(new_X_test)[0]
        print(new_y_pred)
        if new_y_pred == 0:
            result = "Not Spam motherfucker"
        else:
            result = "Nah spam o, run for your life"

        return {'Result': result}


api.add_resource(Spam, '/')
app.run(port=5000, debug=True)
