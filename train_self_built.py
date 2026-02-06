import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Import mô hình tự build
from app.self_built_model import SimpleTransformer, tokenizer  # tokenizer đang dùng BertTokenizer

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5          # Có thể tăng lên nếu có GPU tốt
LEARNING_RATE = 1e-4
MAX_LEN = 128       # Giới hạn độ dài câu
NUM_CLASSES = 2     # Binary: 0=neg, 1=pos

# Sửa mô hình để phù hợp binary classification
class SimpleTransformer(nn.Module): 
    def __init__(self, vocab_size=30522, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        if attention_mask is not None:
            attention_mask = (attention_mask == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=0)
        logits = self.fc(x)
        return logits

model = SimpleTransformer().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Load dataset IMDB
dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']

class IMDBDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text']
        self.labels = data['label']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Training loop
def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation
def evaluate():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    return acc

# Train!
print("Bắt đầu training trên IMDB...")
for epoch in range(EPOCHS):
    train_loss = train_epoch()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")
    acc = evaluate()
    print(f"Test Accuracy: {acc*100:.2f}%")

# Save model sau khi train xong
torch.save(model.state_dict(), "self_built_sentiment_imdb.pt")
print("Đã lưu model vào self_built_sentiment_imdb.pt")