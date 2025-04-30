# Spam-Mail-Prediction-using-Machine-Learning-with-Python

Spam Mail Prediction Using Machine Learning
This project presents a complete machine learning pipeline to detect spam messages using Natural Language Processing (NLP) and classification algorithms. It simulates a real-world text classification task and demonstrates hands-on skills in data preprocessing, feature extraction (TF-IDF), and model evaluation.
________________________________________
Objective
To build an accurate binary classification model that identifies whether a given SMS message is spam or ham (not spam). This task directly applies to industry use cases in content moderation, email filtering, and automated messaging systems.
________________________________________
Tools & Technologies Used
Area	Tools & Libraries
Programming Language	Python 3.x
Data Processing	Pandas, NumPy
NLP & Feature Extraction	Scikit-learn's TfidfVectorizer
ML Model	Naive Bayes (MultinomialNB)
Evaluation	Accuracy, Confusion Matrix, Precision, Recall, F1-score
Visualization (optional)	Seaborn, Matplotlib
________________________________________
 Project Workflow
1. Data Preprocessing
•	Loaded the dataset and dropped irrelevant columns.
•	Renamed columns to text and label.
•	Converted class labels to binary format: spam = 1, ham = 0.
•	Split data into training and testing sets (80/20 split).
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
________________________________________
2. Feature Engineering with TF-IDF
•	Used TfidfVectorizer to transform text into numeric features.
•	Removed English stop words and applied lowercasing for normalization.
tfidf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
Why TF-IDF?
It reduces the influence of common words and increases the weight of informative terms, improving the model’s discriminative power.
________________________________________
3. Model Training
•	Applied Multinomial Naive Bayes: a fast and effective model for text classification tasks involving word counts or TF-IDF scores.
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
________________________________________
4. Model Evaluation
Used standard metrics to assess classification performance:
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
•	Accuracy: ~97%+
•	Precision (Spam): High — good at minimizing false positives
•	Recall (Spam): High — good at catching most spam
•	F1-Score: Balanced measure of precision and recall
________________________________________
Key Takeaways 
•	Demonstrates applied Natural Language Processing with TF-IDF.
•	Implements end-to-end machine learning workflow: from raw text to prediction.
•	Shows understanding of model evaluation beyond accuracy, crucial for imbalanced datasets.
•	Uses domain-appropriate model (Naive Bayes), explaining model choice based on data type and task.
•	Prepared for scaling and deployment (e.g., could be extended into a Flask/Streamlit app).
________________________________________
Future Improvements
•	Text preprocessing: stemming, lemmatization, punctuation removal
•	Use of alternative models: Logistic Regression, SVM, or BERT
•	Real-time detection via API (Flask/Streamlit)
•	Deploy to Hugging Face, Render, or Heroku
