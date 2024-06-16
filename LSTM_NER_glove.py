import numpy as np
import json

# Import necessary libraries
import torch

# Check if GPU is available
gpu_available = torch.cuda.is_available()

if gpu_available:
    # If GPU is available, set device to GPU
    device = torch.device('cuda')
    print("GPU is available")
else:
    # If GPU is not available, set device to CPU
    device = torch.device('cpu')
    print("No GPU detected, using CPU instead")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

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
    tokenized_text = text.split()   
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
    text = value['text']
    label_seq = value['labels']
    tokenized_text = text.split()  
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
test_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    test_data.append(pair)

data = data_train
tokenized_texts = []
labels = []

for key, value in data.items():
    text = value['text']
    label_seq = value['labels']
    tokenized_text = text.split()
    tokenized_texts.append(tokenized_text)
    labels.append(label_seq)
    
train_data = []

for i in range(len(labels)):
    pair = [tokenized_texts[i], labels[i]]
    train_data.append(pair)

import gensim
from gensim.models import KeyedVectors

def load_embeddings_from_json(filename):
    with open(filename, 'r') as file:
        embeddings_dict = json.load(file)
    return embeddings_dict

embeddings_train = 'NER_train_glove.json'
embeddings_dict = load_embeddings_from_json(embeddings_train)

embeddings_test = 'NER_test_glove.json'
embeddings_dict_test = load_embeddings_from_json(embeddings_test)

embeddings_val = 'NER_val_glove.json'
embeddings_dict_val = load_embeddings_from_json(embeddings_val)

def tokens_to_embeddings(tokens, embeddings_dict):
    embeddings = []
    for token in tokens:
        if token in embeddings_dict:
            embeddings.append(embeddings_dict[token])
        else:
            embeddings.append(np.zeros(300))
    return np.array(embeddings)

def labels_to_indices(labels, tag_to_ix):
    return [tag_to_ix[label] for label in labels]

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

embedding_dim = 300
hidden_dim = 128
output_dim = 27

tag_to_ix = {}
for tag_sen in labels:
    for tag in tag_sen:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

print(tag_to_ix)

model = LSTMModel(embedding_dim, hidden_dim, output_dim)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

train_losses = []
val_losses = []

train_f1_scores = []
val_f1_scores = []

num_epochs = 10
for epoch in range(10):
    model.train().to(device)
    total_loss = 0
    predictions = []
    targets = []
    print("running")
    for tokens, labels in train_data:
        token_embeddings = tokens_to_embeddings(tokens, embeddings_dict)
        label_indices = labels_to_indices(labels, tag_to_ix)
        for token_emb, label_idx in zip(token_embeddings, label_indices):
            token_emb = torch.tensor(token_emb).float().unsqueeze(0).to(device)  
            label_idx = torch.tensor(label_idx).long().to(device)  
            optimizer.zero_grad()
            output = model(token_emb).to(device)
            loss = criterion(output, label_idx).to(device)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.append(output.argmax().item())
            targets.append(label_idx.item())
    train_losses.append(total_loss)
    train_f1 = f1_score(targets, predictions, average='macro')
    train_f1_scores.append(train_f1)
    print(f'Epoch {epoch} train loss: {total_loss}, train f1 score: {train_f1}')

    model.eval().to(device)
    total_val_loss = 0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for tokens, labels in val_data:
            token_embeddings = tokens_to_embeddings(tokens, embeddings_dict_val)
            label_indices = labels_to_indices(labels, tag_to_ix)
            for token_emb, label_idx in zip(token_embeddings, label_indices):
                token_emb = torch.tensor(token_emb).float().unsqueeze(0).to(device)  
                label_idx = torch.tensor(label_idx).long().to(device)  
                output = model(token_emb).to(device)
                loss = criterion(output, label_idx).to(device)
                total_val_loss += loss.item()
                val_predictions.append(output.argmax().item())
                val_targets.append(label_idx.item())
    val_losses.append(total_val_loss)
    val_f1 = f1_score(val_targets, val_predictions, average='macro')
    val_f1_scores.append(val_f1)
    print(f'Epoch {epoch} val loss: {total_val_loss}, val f1 score: {val_f1}')

torch.save(model.state_dict(), 'LSTM_NER_glove.pt')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_f1_scores, label='Training Macro-F1')
plt.plot(range(1, num_epochs+1), val_f1_scores, label='Validation Macro-F1')
plt.xlabel('Epoch')
plt.ylabel('Macro-F1 Score')
plt.title('F1 Plot')
plt.legend()

plt.tight_layout()
plt.savefig('LSTM_NER_glove.pdf')
plt.show()



model = LSTMModel(embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('LSTM_NER_glove.pt'))
model.eval()

test_predictions = []
test_targets = []

with torch.no_grad():
    for tokens, labels in test_data:
        token_embeddings = tokens_to_embeddings(tokens, embeddings_dict_test)
        label_indices = labels_to_indices(labels, tag_to_ix)
        for token_emb, label_idx in zip(token_embeddings, label_indices):
            token_emb = torch.tensor(token_emb).float().unsqueeze(0)  
            label_idx = torch.tensor(label_idx).long()  
            output = model(token_emb)
            test_predictions.append(output.argmax().item())
            test_targets.append(label_idx.item())

test_f1 = f1_score(test_targets, test_predictions, average='macro')
print(f'Test Macro-F1: {test_f1}')
#print overall accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_targets, test_predictions)
print(f'Accuracy: {accuracy}')