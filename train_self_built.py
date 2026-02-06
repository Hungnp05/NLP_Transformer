import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Import các thành phần cần từ file model (để tránh trùng lặp định nghĩa class)
from app.self_built_model import SimpleTransformer, tokenizer, DEVICE, PositionalEncoding

# Config
BATCH_SIZE = 32
EPOCHS = 5          # Có thể tăng nếu có GPU tốt
LEARNING_RATE = 1e-4
MAX_LEN = 128       # Giới hạn độ dài câu
NUM_CLASSES = 2     # Binary: 0=neg, 1=pos (dễ train hơn)


# Dataset wrapper cho IMDB
class IMDBDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text']
        self.labels = data['label']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Training functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    return acc

# Main training logic
if __name__ == '__main__':
    print("Bắt đầu training trên IMDB...")

    # Khởi tạo model, optimizer, loss
    model = SimpleTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Load dataset
    print("Đang tải dataset IMDB...")
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']

    train_dataset = IMDBDataset(train_data)
    test_dataset = IMDBDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

        acc = evaluate(model, test_loader, DEVICE)
        print(f"Test Accuracy: {acc*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), "self_built_sentiment_imdb.pt")
    print("Đã lưu model vào self_built_sentiment_imdb.pt")