import numpy as np
import json

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

embeddings_train = 'ATE_train_glove.json'
embeddings_dict = load_embeddings_from_json(embeddings_train)

embeddings_test = 'ATE_test_glove.json'
embeddings_dict_test = load_embeddings_from_json(embeddings_test)

embeddings_val = 'ATE_val_glove.json'
embeddings_dict_val = load_embeddings_from_json(embeddings_val)


label_to_idx = {'O': 0, 'B': 1, 'I': 2}  
num_classes = len(label_to_idx)

def tokens_to_embeddings(tokens, embeddings_dict):
    embeddings = []
    for token in tokens:
        if token in embeddings_dict:
            embeddings.append(embeddings_dict[token])
        else:
            embeddings.append(np.zeros(300))
    return np.array(embeddings)

def labels_to_indices(labels, label_to_idx):
    return [label_to_idx[label] for label in labels]

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

embedding_dim = 300
hidden_dim = 256
output_dim = 3

model = VanillaRNN(embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

train_losses = []
val_losses = []

train_f1_scores = []
val_f1_scores = []
'''
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    for tokens, labels in train_data:
        token_embeddings = tokens_to_embeddings(tokens, embeddings_dict)
        label_indices = labels_to_indices(labels, label_to_idx)
        for token_emb, label_idx in zip(token_embeddings, label_indices):
            token_emb = torch.tensor(token_emb).float().unsqueeze(0)
            label_idx = torch.tensor(label_idx).long()
            optimizer.zero_grad()
            output = model(token_emb)
            loss = criterion(output, label_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions.append(output.argmax().item())
            targets.append(label_idx.item())

    train_losses.append(total_loss / len(train_data))
    train_f1 = f1_score(targets, predictions, average='macro')
    train_f1_scores.append(f1_score(targets, predictions, average='macro'))
    print(f'Epoch {epoch} train loss: {total_loss}, train f1 score: {train_f1}')

    model.eval()
    total_val_loss = 0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for tokens, labels in val_data:
            token_embeddings = tokens_to_embeddings(tokens, embeddings_dict_val)
            label_indices = labels_to_indices(labels, label_to_idx)
            for token_emb, label_idx in zip(token_embeddings, label_indices):
                token_emb = torch.tensor(token_emb).float().unsqueeze(0)  
                label_idx = torch.tensor(label_idx).long() 
                output = model(token_emb)
                loss = criterion(output, label_idx)
                total_val_loss += loss.item()
                val_predictions.append(output.argmax().item())
                val_targets.append(label_idx.item())
    val_losses.append(total_val_loss)
    val_f1 = f1_score(val_targets, val_predictions, average='macro')
    val_f1_scores.append(val_f1)
    print(f'Epoch {epoch} val loss: {total_val_loss}, val f1 score: {val_f1}')

torch.save(model.state_dict(), 'RNN_ATE_glove.pt')

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
plt.show()

plt.savefig('RNN_ATE_glove.png')
'''

model = VanillaRNN(embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('RNN_ATE_glove.pt'))
model.eval()

test_predictions = []
test_targets = []
for tokens, labels in test_data:
    token_embeddings = tokens_to_embeddings(tokens, embeddings_dict_test)
    label_indices = labels_to_indices(labels, label_to_idx)
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









