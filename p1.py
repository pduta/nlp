import string 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer 
 
nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('wordnet') 
 
def preprocess_corpus(text): 
    # 1. Conversion to Lowercase 
    text = text.lower() 
 
    # 2. Removal of Punctuation 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
 
    # 3. Tokenization 
    tokens = word_tokenize(text) 
 
    # 4. Filtration of Stop Words 
    stop_words = set(stopwords.words('english')) 
    filtered_tokens = [w for w in tokens if w not in stop_words] 
 
    # 5. Stemming (Reducing to root form, e.g., "running" -> "run") 
    stemmer = PorterStemmer() 
    stemmed = [stemmer.stem(w) for w in filtered_tokens] 
 
    # 6. Lemmatization (Context-aware base form, e.g., "better" -> "good") 
    lemmatizer = WordNetLemmatizer() 
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in filtered_tokens] 
 
    return { 
        "tokens": tokens, 
        "stemmed": stemmed, 
        "lemmatized": lemmatized 
    } 
 
sample_text = "The quick brown foxes are jumping over the lazy dogs!" 
print(preprocess_corpus(sample_text)) 
