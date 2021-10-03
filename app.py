from flask import Flask,render_template,request
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np
import os
import pickle

###Loading model and cv
model = load_model('model.h5')

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r"xfxfxxb", '', text) # remove numbers

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text) 
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        
        otherData = pd.DataFrame()
        otherData['text'] = request.form['review']
        otherData['text_clean'] = otherData['text'].apply(cleaningText)
        otherData['text_clean'] = otherData['text_clean'].apply(casefoldingText)
        otherData.drop(['text'], axis = 1, inplace = True)

        otherData['text_preprocessed'] = otherData['text_clean'].apply(tokenizingText)
        otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(filteringText)
        otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(stemmingText)
       

        X_otherData = otherData['text_preprocessed'].apply(toSentence)
        X_otherData = tokenizer.texts_to_sequences(X_otherData.values)
        X_otherData = pad_sequences(X_otherData, maxlen = X.shape[1])
        

        y_pred_otherData = model.predict(X_otherData)
       

        return render_template('result.html',prediction=y_pred_otherData)

if __name__ == "__main__":
    app.run(debug=True)    
