import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Định nghĩa DEVICE (dùng CPU để tránh lỗi device mismatch trên máy không có GPU)
DEVICE = torch.device("cpu")
# Nếu máy có GPU và muốn dùng: 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer (dùng bert-base-uncased để khớp vocab_size=30522)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=30522, d_model=256, nhead=4, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        src_key_padding_mask = None
        if attention_mask is not None:
            # attention_mask từ tokenizer là 1=keep, 0=pad
            # src_key_padding_mask cần True=masked (pad), False=keep
            src_key_padding_mask = (attention_mask == 0)  # (batch, seq_len)

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=0)                                 # mean pooling: (batch, d_model)
        logits = self.fc(x)                               # (batch, num_classes)
        return logits

# Khởi tạo model
self_built_model = SimpleTransformer(num_classes=2)
self_built_model.to(DEVICE)
self_built_model.eval()

self_built_model.load_state_dict(
    torch.load(r"E:\code\NLP\self_built_sentiment_imdb.pt", map_location=DEVICE)
)

# Nếu đã train và có file model, uncomment dòng này:
# self_built_model.load_state_dict(torch.load("self_built_sentiment_imdb.pt", map_location=DEVICE))

# Hàm predict
def predict_self_built(text: str):
    """
    Dự đoán sentiment cho văn bản đầu vào.
    Hiện tại model dùng weights random → kết quả random.
    Sau khi train, kết quả sẽ chính xác hơn.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        logits = self_built_model(input_ids, attention_mask)

    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
    predicted_class = int(torch.argmax(logits, dim=-1).item())

    # Map class sang label
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    label_text = labels[predicted_class]

    # Tính vector embedding (mean pooling của hidden states)
    with torch.no_grad():
        x = self_built_model.embedding(input_ids) * math.sqrt(self_built_model.embedding.embedding_dim)
        x = self_built_model.pos_encoder(x)
        x = x.permute(1, 0, 2)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0).to(DEVICE)

        hidden = self_built_model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        vector = hidden.mean(dim=0).squeeze().cpu().numpy().tolist()[:10]  # 10 phần tử đầu

    return {
        "prediction": f"Self-built Transformer - {label_text} (class {predicted_class})",
        "probabilities": [round(p, 4) for p in probabilities],
        "vector": vector
    }