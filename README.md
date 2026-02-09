# 📰 Fake News Detection using Machine Learning

This project implements a **Fake News Detection system** using **Python, TF-IDF Vectorizer, and Logistic Regression**.  
It uses a real-world dataset from Kaggle that contains separate files for fake and real news articles. The model is trained to classify news text as **Fake** or **Real**.

---

## 📁 Dataset

**Source (Kaggle):**  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains two files:
- `Fake.csv` → Fake news articles  
- `True.csv` → Real news articles  

Each file contains a `text` column with the news content.

Labels are created manually:
- Fake = 1  
- Real = 0  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## 🧠 Approach

1. Load `Fake.csv` and `True.csv`
2. Add labels to each dataset (Fake = 1, Real = 0)
3. Combine and shuffle the datasets
4. Split data into training and testing sets
5. Convert text to numerical features using **TF-IDF Vectorizer (with unigrams and bigrams)**
6. Train a **Logistic Regression** classifier
7. Evaluate the model using:
   - Accuracy
   - Classification Report
   - Confusion Matrix
8. Add a function to predict custom news text

---

## 📊 Model Evaluation

The model prints:
- Accuracy score  
- Precision, Recall, F1-score (Classification Report)  
- Confusion Matrix  

These metrics help evaluate how well the model distinguishes between fake and real news.

---

## ▶️ How to Run the Project

1. Clone the repository:
git clone https://github.com/Itsayaanafsar/Fake_News_Detector.git

2. Install dependencies:
   pip install pandas numpy scikit-learn

3. Make sure these files are in the same folder:
  Fake.csv
  True.csv
  fake_news_detector.py

4. Run the script:
   python fake_news_detector.py

5.You will see:
  -Model accuracy
  -Classification report
  -Confusion matrix
  -Prediction for a sample news text 

---

## 📂 Project Structure
fake-news-detector/
├── Fake.csv
├── True.csv
├── fake_news_detector.py
└── README.md

---

## 🎯 What This Project Demonstrates
Text preprocessing using TF-IDF

-Binary classification using Logistic Regression
-Working with real-world datasets
-Model evaluation using standard ML metrics
-Building a simple prediction function for user input

