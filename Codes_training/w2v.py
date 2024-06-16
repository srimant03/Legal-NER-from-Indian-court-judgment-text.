import numpy as np
import json

filename = 'NER_train.json'

with open(filename, 'r') as file:
    data_train = json.load(file)

filename = 'NER_test.json'

with open(filename, 'r') as file:
    data_test = json.load(file)
    
filename = 'NER_val.json'

with open(filename, 'r') as file:
    data_val = json.load(file)

data = data_val
tokenized_texts = []
labels = []

for key, value in data.items():
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  
    
#     Store the tokenized text
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
val_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    val_data.append(pair)

data = data_test
tokenized_texts = []
labels = []

for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    tokenized_text = text.split()  
    
#     Store the tokenized text
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
test_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    test_data.append(pair)

data = data_train
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
train_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    train_data.append(pair)

import gensim
from gensim.models import KeyedVectors

word_to_vec_map = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

train_word_embeddings = {}
val_word_embeddings = {}
test_word_embeddings = {}

for i in range(len(train_data)):
    tokenized_text = train_data[i][0]
    for token in tokenized_text:
        if token in word_to_vec_map:
            train_word_embeddings[token] = word_to_vec_map[token]
        else:
            train_word_embeddings[token] = [0]*300

for i in range(len(val_data)):
    tokenized_text = val_data[i][0]
    for token in tokenized_text:
        if token in word_to_vec_map:
            val_word_embeddings[token] = word_to_vec_map[token]
        else:
            val_word_embeddings[token] = [0]*300

for i in range(len(test_data)):
    tokenized_text = test_data[i][0]
    for token in tokenized_text:
        if token in word_to_vec_map:
            test_word_embeddings[token] = word_to_vec_map[token]
        else:
            test_word_embeddings[token] = [0]*300

for token, word_embedding in train_word_embeddings.items():
    if isinstance(word_embedding, np.ndarray):
        train_word_embeddings[token] = word_embedding.tolist()

for token, word_embedding in val_word_embeddings.items():
    if isinstance(word_embedding, np.ndarray):
        val_word_embeddings[token] = word_embedding.tolist()

for token, word_embedding in test_word_embeddings.items():
    if isinstance(word_embedding, np.ndarray):
        test_word_embeddings[token] = word_embedding.tolist()
        
with open('NER_train_W2V.json', 'w') as file:
    json.dump(train_word_embeddings, file)

with open('NER_val_W2V.json', 'w') as file:
    json.dump(val_word_embeddings, file)

with open('NER_test_W2V.json', 'w') as file:
    json.dump(test_word_embeddings, file)