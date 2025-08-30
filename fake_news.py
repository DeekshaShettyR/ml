import os
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("Starting Fake News Training...")

os.makedirs('models', exist_ok=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load dataset
df = pd.read_csv('dataset/fake_news.csv').fillna('')
df['content'] = df['title1_en'] + ' ' + df['title2_en']
df['label'] = df['label'].map({'unrelated': 0, 'agreed': 1})
df = df.dropna(subset=['label'])

# Text preprocessing
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['content'] = df['content'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['content']).toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open('models/fake_news_model.pkl', 'wb'))
pickle.dump(tfidf, open('models/tfidf_vectorizer.pkl', 'wb'))
print("Model and vectorizer saved in 'models/' folder.")
