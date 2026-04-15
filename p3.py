import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

text = """
Python is a powerful programming language. Python is widely used in data science,
machine learning, and web development. Many developers love Python because Python
is easy to learn and Python has a large community of developers who contribute to
Python projects and Python libraries every day.
"""

stop_words = set(stopwords.words('english'))

# Extract words (lowercase, letters only)
words = re.findall(r'\b[a-z]+\b', text.lower())

# Filter out stopwords
filtered = [w for w in words if w not in stop_words]

# Count and display
word_freq = Counter(filtered)

print(f"{'Word':<20} {'Frequency':<10} {'Bar'}")
print("-" * 45)
for word, freq in word_freq.most_common(10):
    bar = "█" * freq
    print(f"{word:<20} {freq:<10} {bar}")
