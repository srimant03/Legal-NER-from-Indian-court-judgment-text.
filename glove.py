import json
import torch
import gensim
import numpy as np

torch.manual_seed(1)

filename = 'ATE_train.json'

with open(filename, 'r') as file:
    data_train = json.load(file)

filename = 'ATE_test.json'

with open(filename, 'r') as file:
    data_test = json.load(file)
    
filename = 'ATE_val.json'

with open(filename, 'r') as file:
    data_val = json.load(file)

data = data_val
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  # Using simple split for illustration
    
    # Store the tokenized text
    tokenized_texts.append(tokenized_text)
    # Store the labels
    labels.append(label_seq)
    
val_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    val_data.append(pair)

data = data_test
tokenized_texts = []
labels = []

# Iterate over the items in the parsed JSON object
for key, value in data.items():
    # Extract text and labels for each item
    text = value['text']
    label_seq = value['labels']
    
    # Tokenize the text (if needed)
    tokenized_text = text.split()  # Using simple split for illustration
    
    # Store the tokenized text
    tokenized_texts.append(tokenized_text)
    # Store the labels
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
    tokenized_text = text.split()  # Using simple split for illustration
    
    # Store the tokenized text
    tokenized_texts.append(tokenized_text)
    # Store the labels
    labels.append(label_seq)
    
train_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    train_data.append(pair)

from gensim.scripts.glove2word2vec import glove2word2vec

# Convert GloVe format to word2vec format
glove_input_file = 'glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Load GloVe embeddings in word2vec format
glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


train_word_embeddings = {}
val_word_embeddings = {}
test_word_embeddings = {}

for i in range(len(train_data)):
    tokenized_text = train_data[i][0]
    for token in tokenized_text:
        if token in glove_model:
            train_word_embeddings[token] = glove_model[token]
        else:
            train_word_embeddings[token] = [0]*300

for i in range(len(val_data)):
    tokenized_text = val_data[i][0]
    for token in tokenized_text:
        if token in glove_model:
            val_word_embeddings[token] = glove_model[token]
        else:
            val_word_embeddings[token] = [0]*300

for i in range(len(test_data)):
    tokenized_text = test_data[i][0]
    for token in tokenized_text:
        if token in glove_model:
            test_word_embeddings[token] = glove_model[token]
        else:
            test_word_embeddings[token] = [0]*300

for token, word_embedding in train_word_embeddings.items():
    #check if the word embedding is a numpy array
    if isinstance(word_embedding, np.ndarray):
        #convert it to a list
        train_word_embeddings[token] = word_embedding.tolist()

for token, word_embedding in val_word_embeddings.items():
    if isinstance(word_embedding, np.ndarray):
        val_word_embeddings[token] = word_embedding.tolist()

for token, word_embedding in test_word_embeddings.items():
    if isinstance(word_embedding, np.ndarray):
        test_word_embeddings[token] = word_embedding.tolist()

with open('ATE_train_glove.json', 'w') as f:
    json.dump(train_word_embeddings, f)

with open('ATE_val_glove.json', 'w') as f:
    json.dump(val_word_embeddings, f)

with open('ATE_test_glove.json', 'w') as f:
    json.dump(test_word_embeddings, f)

