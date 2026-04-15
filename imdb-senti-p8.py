import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
# 1. Load dataset (update path to your downloaded file) 
df = pd.read_csv('IMDB Dataset.csv') 
# 2. Split into Training and Testing sets 
X_train, X_test, y_train, y_test = train_test_split(df['review'], 
df['sentiment'], test_size=0.2, random_state=42) 
# 3. Vectorize text data 
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7) 
X_train_vector = tfidf.fit_transform(X_train) 
X_test_vector = tfidf.transform(X_test) 
# 4. Train Model 
model = LogisticRegression() 
model.fit(X_train_vector, y_train) 
# 5. Evaluate 
predictions = model.predict(X_test_vector) 
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
