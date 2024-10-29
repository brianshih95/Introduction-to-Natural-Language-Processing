import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


LABELS = ['ineffective', 'unnecessary', 'pharma', 'rushed', 'side-effect',
          'mandatory', 'country', 'ingredients', 'political', 'none',
          'conspiracy', 'religious']
MAX_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 8
LR = 5e-5


class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tweet = item['tweet']
        label_vector = torch.zeros(len(LABELS))
        if 'labels' in item:
            for concern in item['labels']:
                if concern in LABELS:
                    label_vector[LABELS.index(concern)] = 1

        encoding = self.tokenizer.encode_plus(
            tweet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }


class Bert(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = x.pooler_output
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def train(model, train_loader, val_loader, device, criterion, optimizer):
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                val_loss += loss.item()

                predictions = (output >= 0.5).float()
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)

        correct = 0
        for prediction, label in zip(val_predictions, val_labels):
            if np.array_equal(prediction, label):
                correct += 1
        acc = correct / len(val_predictions)

        print(f'Training Loss: {train_loss:.3f}')
        print(f'Validation Loss: {val_loss:.3f}')
        print(f'Validation Accuracy: {acc:.3f}\n\n')


def evaluate_categories(model, val_loader, device):
    model.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating categories'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask)

            predictions = (output >= 0.5).float()
            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels, val_predictions, average=None)
    performance = pd.DataFrame({
        'Category': LABELS,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
    })

    print("\nCategory Performance (Sorted by F1-Score):")
    print(performance.sort_values('F1-Score').to_string(index=False))

    errors = (val_predictions != val_labels)
    error_matrix = np.zeros((len(LABELS), len(LABELS)))

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j:
                error_matrix[i, j] = np.sum(errors[:, i] & errors[:, j])

    plt.figure(figsize=(12, 10))
    sns.heatmap(error_matrix, xticklabels=LABELS, yticklabels=LABELS,
                annot=True, fmt='g', cmap='YlOrRd')
    plt.title('Error Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig('error_cooccurrence.png')
    plt.close()


def predict(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids, attention_mask)

            predictions.extend((output >= 0.5).float().cpu().numpy())

    return predictions


train_data = load_data('data/train.json')
val_data = load_data('data/val.json')
test_data = load_data('data/test.json')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = TweetDataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = TweetDataset(val_data, tokenizer, MAX_LENGTH)
test_dataset = TweetDataset(test_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Bert(len(LABELS)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

train(model, train_loader, val_loader, device, criterion, optimizer)
evaluate_categories(model, val_loader, device)
predictions = predict(model, test_loader, device)

submission = pd.DataFrame(predictions, columns=LABELS)
submission.insert(0, 'index', range(len(predictions)))
submission.to_csv('submission.csv', index=False)
