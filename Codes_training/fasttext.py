import json
import torch
import gensim
import numpy as np

torch.manual_seed(1)

# Load fastText word embeddings
fasttext_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)

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


# Load fastText word embeddings
fasttext_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)

#now use this fasttext_model to create word embeddings for the tokens in the dataset
#for train, test and val datasets
#also save these embeddings for future use
#basically store each token and its corresponding word embedding in a dictionary and save it to a json file

train_word_embeddings = {}
val_word_embeddings = {}
test_word_embeddings = {}

for i in range(len(train_data)):
    tokenized_text = train_data[i][0]
    for token in tokenized_text:
        if token in fasttext_model:
            train_word_embeddings[token] = fasttext_model[token]
        else:
            train_word_embeddings[token] = [0]*300


for i in range(len(val_data)):
    tokenized_text = val_data[i][0]
    for token in tokenized_text:
        if token in fasttext_model:
            val_word_embeddings[token] = fasttext_model[token]
        else:
            val_word_embeddings[token] = [0]*300

for i in range(len(test_data)):
    tokenized_text = test_data[i][0]
    for token in tokenized_text:
        if token in fasttext_model:
            test_word_embeddings[token] = fasttext_model[token]
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

#save these dictionaries to a json file
with open('ATE_train_fasttext.json', 'w') as file:
    json.dump(train_word_embeddings, file)

with open('ATE_val_fasttext.json', 'w') as file:
    json.dump(val_word_embeddings, file)

with open('ATE_test_fasttext.json', 'w') as file:
    json.dump(test_word_embeddings, file)




