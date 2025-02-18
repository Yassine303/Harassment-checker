import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from transformers import pipeline

# Load dataset (adjust path accordingly)
df = pd.read_csv("labeled_data.csv")

# Keep only necessary columns
df = df[['comment_text', 'toxic']]

# Convert labels: 1 = toxic, 0 = non-toxic
df['toxic'] = df['toxic'].apply(lambda x: 'harassment' if x == 1 else 'neutral')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['comment_text'] = df['comment_text'].apply(clean_text)

# Show sample
print(df.head())



# Split data
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




# Load pre-trained model
harassment_detector = pipeline("text-classification", model="unitary/toxic-bert")

# Test the model
text = "You are the worst! No one likes you."
result = harassment_detector(text)

print(result)  # Output: [{'label': 'toxic', 'score': 0.98}]

