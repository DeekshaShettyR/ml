from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('models/fake_news_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title1 = request.form.get('title1', '').strip()
    title2 = request.form.get('title2', '').strip()

    if not title1 or not title2:
        return render_template('index.html', prediction_text="Please enter both title1 and title2.")

    combined_text = title1 + ' ' + title2
    clean_news = clean_text(combined_text)
    vect = tfidf.transform([clean_news]).toarray()
    prediction = model.predict(vect)[0]
    result = "Real" if prediction == 1 else "Fake"
    return render_template('index.html', prediction_text=f'The news is {result}')

if __name__ == '__main__':
    app.run(debug=True)
