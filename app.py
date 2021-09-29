model = tf.keras.models.load_model('model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        otherData = pd.DataFrame()
        otherData['text'] = ['review']
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
        otherData['Result Prediction'] = y_pred_otherData


        return render_template('result.html',prediction=otherData)

if __name__ == "__main__":
    app.run(debug=True)  
