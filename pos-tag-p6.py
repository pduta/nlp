import nltk 
from collections import Counter 
 
# Download necessary data packages 
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
 
def process_text(file_path): 
    try: 
        with open(file_path, 'r', encoding='utf-8') as f: 
            text = f.read() 
 
        # Tokenize and Tag 
        tokens = nltk.word_tokenize(text) 
        tagged_words = nltk.pos_tag(tokens) 
 
        # 1. Extract all Nouns 
        # NN: singular, NNS: plural, NNP: proper singular, NNPS: proper 
plural 
        nouns = [word for word, tag in tagged_words if tag.startswith('NN')] 
 
        # 2. Count POS frequencies 
        pos_counts = Counter(tag for word, tag in tagged_words) 
 
        # Output Results 
print("--- EXTRACTED NOUNS ---") 
print(", ".join(nouns[:20]), "..." if len(nouns) > 20 else "") 
print(f"\nTotal Nouns Found: {len(nouns)}") 
print("\n--- POS FREQUENCY DICTIONARY ---") 
print(dict(pos_counts)) 
except FileNotFoundError: 
print("Error: The file was not found.") 
if __name__ == "__main__": 
# Replace 'yourfile.txt' with your actual filename 
process_text('sample.txt')
