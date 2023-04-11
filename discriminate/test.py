import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = ElectraForSequenceClassification.from_pretrained('discriminate/dis/model')
model.to(device)
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')


# Inference on a single sentence at a time
def predict(sentence):
    encoding = tokenizer(sentence, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1)
    return pred.item()
