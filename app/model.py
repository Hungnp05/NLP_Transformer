from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Quan trọng: bật output_hidden_states
model = BertForSequenceClassification.from_pretrained(
    model_name,
    output_hidden_states=True
)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    predicted_class = np.argmax(probabilities)
    
    # Lấy vector từ hidden state cuối cùng, token [CLS]
    if outputs.hidden_states is not None:
        cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy().tolist()
        vector_display = cls_embedding[:10]  # hiển thị 10 phần tử đầu
    else:
        vector_display = ["(hidden states not available)"]

    return {
        "prediction": f"Sentiment class: {predicted_class} (0: very negative, 4: very positive)",
        "probabilities": [round(p, 4) for p in probabilities],  # làm đẹp số
        "vector": vector_display
    }