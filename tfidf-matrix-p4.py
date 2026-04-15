from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd 
 
# 1. Prepare your document set 
corpus = [ 
    'Ubuntu is a popular Linux distribution.', 
    'Python is great for data science.', 
    'The TF-IDF matrix helps in text processing.', 
    'Ubuntu and Python work well together.' 
] 
 
# 2. Initialize and fit the vectorizer 
vectorizer = TfidfVectorizer() 
tfidf_matrix = vectorizer.fit_transform(corpus) 
 
# 3. Convert to a readable format (DataFrame) 
feature_names = vectorizer.get_feature_names_out() 
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names) 
print(df) 
