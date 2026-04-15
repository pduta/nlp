import tensorflow as tf 
import numpy as np 
import os 
# 1. Load Dataset 
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
'https://storage.googleapis.com') 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') 
# 2. Vectorize the text 
vocab = sorted(set(text)) 
char2idx = {u:i for i, u in enumerate(vocab)} 
idx2char = np.array(vocab) 
text_as_int = np.array([char2idx[c] for c in text]) 
# 3. Create Training Examples 
seq_length = 100 
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) 
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) 
def split_input_target(chunk): 
input_text = chunk[:-1] 
target_text = chunk[1:] 
return input_text, target_text 
dataset = sequences.map(split_input_target).shuffle(10000).batch(64, 
drop_remainder=True) 
# 4. Build the Model 
model = tf.keras.Sequential([ 
tf.keras.layers.Embedding(len(vocab), 256, batch_input_shape=[64, None]), 
tf.keras.layers.LSTM(1024, return_sequences=True, stateful=True, 
recurrent_initializer='glorot_uniform'), 
tf.keras.layers.Dense(len(vocab)) 
]) 
# 5. Compile and Train (Simplified) 
model.compile(optimizer='adam', 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)) 
model.fit(dataset, epochs=5) # 5 epochs for a quick demonstration 
