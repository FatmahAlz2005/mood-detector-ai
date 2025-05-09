from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "I am very happy today",
    "This is the best day ever!",
    "I feel sad and tired",
    "I am angry and frustrated",
    "I am feeling okay",
    "I don't feel anything special"
]

labels = ["happy", "happy", "sad", "angry", "neutral", "neutral"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

print("Enter a sentence to detect your mood:")
user_input = input("> ")
user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)
print(f"\nPredicted Mood: {prediction[0]}")
