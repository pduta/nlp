import collections 
import math 
def build_model(corpus): 
# Tokenize and add sentence markers 
tokens = corpus.lower().split() 
vocab = set(tokens) 
V = len(vocab) 
# Counts 
unigram_counts = collections.Counter(tokens) 
bigrams = list(zip(tokens, tokens[1:])) 
bigram_counts = collections.Counter(bigrams) 
return unigram_counts, bigram_counts, V, len(tokens) 
def get_sentence_probability(sentence, unigram_counts, bigram_counts, V, N): 
words = sentence.lower().split() 
log_prob = 0 
# We calculate using log probabilities to prevent numerical underflow 
for i in range(1, len(words)): 
prev_word = words[i-1] 
curr_word = words[i] 
# Bigram Add-One Smoothing formula 
count_prev = unigram_counts.get(prev_word, 0) 
count_bigram = bigram_counts.get((prev_word, curr_word), 0) 
prob = (count_bigram + 1) / (count_prev + V) 
log_prob += math.log(prob) 
return math.exp(log_prob) 
# --- Execution --- 
training_data = "the cat sat on the mat the dog sat on the rug" 
test_sentence = "the cat sat on the rug" 
u_counts, b_counts, v_size, total_tokens = build_model(training_data) 
prob = get_sentence_probability(test_sentence, u_counts, b_counts, v_size, 
total_tokens) 
print(f"Vocabulary Size: {v_size}") 
print(f"Probability of '{test_sentence}': {prob:.10f}") 
