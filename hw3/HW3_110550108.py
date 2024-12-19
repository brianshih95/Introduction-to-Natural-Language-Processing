import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ResponseQualityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        self.utterances = [item['u'] for item in data]
        self.situations = [' '.join(item['s']) for item in data]
        self.responses = [item['r'] for item in data]

        self.is_test = is_test
        if not is_test:
            self.labels = [item['r.label'] for item in data]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        text = f"Question: {self.utterances[idx]} Situations: {self.situations[idx]} Response: {self.responses[idx]}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        if self.is_test:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }


def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)


        model.eval()
        val_losses = []
        val_preds, val_true = [], []

        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Validation", unit="batch")
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                preds = torch.argmax(outputs.logits, dim=1)

                val_losses.append(loss.item())
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_accuracy = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch+1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return model


def main():
    with open('data/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('data/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = ResponseQualityDataset(train_data, tokenizer)
    val_dataset = ResponseQualityDataset(val_data, tokenizer)
    test_dataset = ResponseQualityDataset(test_data, tokenizer, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    trained_model = train_model(model, train_loader, val_loader, device)

    trained_model.eval()
    test_preds = []

    trained_model.load_state_dict(torch.load('best_model.pth'))

    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Predicting", unit="batch")
        for batch in test_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = trained_model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            test_preds.extend(preds.cpu().numpy())

    submission = pd.DataFrame({
        'index': range(len(test_preds)),
        'response_quality': test_preds
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
