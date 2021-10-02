from flask import Flask,render_template,request
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

###Loading model and cv
model = keras.models.load_model ("model.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# def cleaningText(text):
#     text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
#     text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
#     text = re.sub(r'RT[\s]', '', text) # remove RT
#     text = re.sub(r"http\S+", '', text) # remove link
#     text = re.sub(r'[0-9]+', '', text) # remove numbers
#     text = re.sub(r"xfxfxxb", '', text) # remove numbers

#     text = text.replace('\n', ' ') # replace new line into space
#     text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
#     text = text.strip(' ') # remove characters space from both left and right text
#     return text

# def casefoldingText(text): # Converting all the characters in a text into lower case
#     text = text.lower() 
#     return text

# def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
#     text = word_tokenize(text) 
#     return text

# def filteringText(text): # Remove stopwors in a text
#     listStopwords = set(stopwords.words('indonesian'))
#     filtered = []
#     for txt in text:
#         if txt not in listStopwords:
#             filtered.append(txt)
#     text = filtered 
#     return text

# def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()
#     text = [stemmer.stem(word) for word in text]
#     return text

# def toSentence(list_words): # Convert list of words into sentence
#     sentence = ' '.join(word for word in list_words)
#     return sentence

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
       
#         otherData['text'] = ['PPKM diperpanjang saja, enak WFH',
#                      'PPKM diakhiri saja, tidak bisa jalan jalan ke Malang']
#         otherData = pd.DataFrame()
#         otherData['text'] = request.form['review']
#         otherData['text_clean'] = otherData['text'].apply(cleaningText)
#         otherData['text_clean'] = otherData['text_clean'].apply(casefoldingText)
#         otherData.drop(['text'], axis = 1, inplace = True)

#         otherData['text_preprocessed'] = otherData['text_clean'].apply(tokenizingText)
#         otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(filteringText)
#         otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(stemmingText)
#         otherData

#         X_otherData = otherData['text_preprocessed'].apply(toSentence)
#         X_otherData = tokenizer.texts_to_sequences(X_otherData.values)
#         X_otherData = pad_sequences(X_otherData, maxlen = X.shape[1])
#         X_otherData

#         y_pred_otherData = model.predict(X_otherData)
#         pred = model.predict(y_pred_otherData)


#         otherData['Result Prediction'] = y_pred_otherData
#         polarity_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
#         otherData['Result Prediction'] = otherData['Result Prediction'].map(polarity_decode)
#         otherData
        
        
        cv = pickle.load(open('cv.pkl','rb'))
        model = pickle.load(open('review.pkl','rb'))        
        new_review = request.form['review']
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        pred = model.predict(new_X_test)
        return render_template('result.html',prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)    
