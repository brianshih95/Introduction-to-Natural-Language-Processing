import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import json


BATCH_SIZE = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
N_EPOCHS = 40
LR = 2e-5
WEIGHT_DECAY = 1e-6


def toLowerCase(text):
    return text.lower()


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text, verified_purchase, helpful_vote, labels = [
            b.to(device) for b in batch]
        optimizer.zero_grad()
        predictions = model(text, verified_purchase, helpful_vote)
        loss = criterion(predictions, labels)
        acc = (predictions.argmax(1) == labels).float().mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class ReviewDataset(Dataset):
    def __init__(self, text_features, verified_purchase, helpful_vote, ratings=None):
        self.text_features = torch.FloatTensor(text_features)
        self.verified_purchase = torch.FloatTensor(verified_purchase)
        self.helpful_vote = torch.FloatTensor(helpful_vote)
        self.ratings = torch.LongTensor(
            ratings) if ratings is not None else None

    def __len__(self):
        return len(self.text_features)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.text_features[idx], self.verified_purchase[idx], self.helpful_vote[idx], self.ratings[idx] - 1
        else:
            return self.text_features[idx], self.verified_purchase[idx], self.helpful_vote[idx]


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, verified_purchase, helpful_vote):
        x = torch.cat((text, verified_purchase.unsqueeze(1),
                       helpful_vote.unsqueeze(1)), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


with open('data/train.json', 'r') as f:
    train_data = json.load(f)

with open('data/test.json', 'r') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
train_df['processed_text'] = train_df['title'] + " " + train_df['text']
train_df['processed_text'] = train_df['processed_text'].apply(toLowerCase)

test_df = pd.DataFrame(test_data)
test_df['processed_text'] = test_df['title'] + " " + test_df['text']
test_df['processed_text'] = test_df['processed_text'].apply(toLowerCase)

le = LabelEncoder()
train_df['verified_purchase'] = le.fit_transform(train_df['verified_purchase'])
test_df['verified_purchase'] = le.transform(test_df['verified_purchase'])

cv = CountVectorizer(max_features=10000)
X_train = cv.fit_transform(train_df['processed_text']).toarray()
X_test = cv.transform(test_df['processed_text']).toarray()

train_dataset = ReviewDataset(
    X_train, train_df['verified_purchase'].values, train_df['helpful_vote'].values, train_df['rating'].values)
test_dataset = ReviewDataset(
    X_test, test_df['verified_purchase'].values, test_df['helpful_vote'].values)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          num_workers=2, pin_memory=True, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

INPUT_DIM = X_train.shape[1] + 2
OUTPUT_DIM = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    print(
        f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}')


model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        text, verified_purchase, helpful_vote = [b.to(device) for b in batch]
        output = model(text, verified_purchase, helpful_vote)
        predictions.extend(output.argmax(1).cpu().numpy())

predictions = np.array(predictions) + 1

submission = pd.DataFrame({
    'index': ['index_' + str(i) for i in range(len(predictions))],
    'rating': predictions
})
submission.to_csv('submission.csv', index=False)
