import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import pandas as pd


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained Electra tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator')

# Load train and dev data from csv files
train_data = pd.read_csv('train_data.csv')
dev_data = pd.read_csv('dev_data.csv')

# Tokenize train and dev data
train_encodings = tokenizer(list(train_data['sentence']), truncation=True, padding=True)
dev_encodings = tokenizer(list(dev_data['sentence']), truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(list(train_data['label']))
dev_labels = torch.tensor(list(dev_data['label']))


# Create PyTorch DataLoader objects for train and dev data
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
dev_dataset = CustomDataset(dev_encodings, dev_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# Fine-tune Electra model on train data
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from tqdm import tqdm

epochs = 8
for epoch in range(epochs):
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': loss.item()})

model.save_pretrained('dis/model')
