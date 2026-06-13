import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())
# Remove unwanted columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("\nCleaned Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Convert labels to numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and Target
X = df['message']
y = df['label']

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100)

# Test Message

sample_message = ["FREE entry into our contest! Win cash prize now!"]

sample_vector = vectorizer.transform(sample_message)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("\nPrediction: Spam")
else:
    print("\nPrediction: Ham")