import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

fake_df['label'] = 1
true_df['label'] = 0

fake_df = fake_df[['text', 'label']]
true_df = true_df[['text', 'label']]

df = pd.concat([fake_df, true_df], ignore_index=True)
df  = df.sample(frac=1, random_state=3).reset_index(drop=True)

x = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(1,2), dtype=np.float32)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_fake_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Fake News" if pred == 1 else "Real News"

print(predict_fake_news("Breaking: Scientists discover a new planet in our solar system!"))
