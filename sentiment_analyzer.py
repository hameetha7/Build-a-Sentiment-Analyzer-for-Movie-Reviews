import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 1. Load the dataset
df = pd.read_csv("imdb_sample.csv")  # Replace with your actual CSV filename if needed

# 2. Clean the text
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].astype(str).apply(clean_text)

# 3. Encode labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42
)

# 5. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# 7. Test the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%\n")

# 8. Show 5 sample predictions
print("Sample Predictions:")
num_samples = min(5, len(X_test))
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
for idx in sample_indices:
    review = X_test.iloc[idx]
    true_label = "POSITIVE" if y_test.iloc[idx] == 1 else "NEGATIVE"
    pred_label = "POSITIVE" if y_pred[idx] == 1 else "NEGATIVE"
    print(f'Review: "{df["review"].iloc[X_test.index[idx]]}"')
    print(f"Prediction: {pred_label} (Actual: {true_label})\n")

# 9. Input box for custom reviews
def predict_custom_review():
    while True:
        custom_review = input('Enter a movie review (or type "exit" to quit): ')
        if custom_review.lower() == 'exit':
            break
        clean_custom = clean_text(custom_review)
        vec_custom = vectorizer.transform([clean_custom])
        pred = model.predict(vec_custom)[0]
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f'Prediction: {label}')

if __name__ == "__main__":
    print("Try custom reviews like:")
    print('  The movie was boring and too long.')
    print('  Amazing performance and storyline!\n')
    predict_custom_review()