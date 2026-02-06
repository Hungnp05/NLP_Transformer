# Transformer Sentiment Analysis Web App

Ứng dụng web đơn giản sử dụng **FastAPI** để dự đoán cảm xúc (sentiment analysis) từ văn bản đầu vào. Có hai mô hình:

1. **Mô hình pre-trained** (BERT fine-tuned) – chính xác cao, sẵn dùng ngay.
2. **Mô hình tự build & tự train** (Transformer từ scratch bằng PyTorch) – để học và tùy chỉnh.

## Mục đích project

- Minh họa cách tích hợp mô hình Transformer vào web app.
- So sánh mô hình pre-trained (từ Hugging Face) và mô hình tự code + tự train.
- Hiển thị dự đoán, xác suất, và vector embedding để dễ hiểu nội bộ mô hình.


## Yêu cầu hệ thống & Cài đặt

### Yêu cầu
- Python 3.8+
- Máy có GPU (khuyến nghị) để train nhanh hơn, nhưng CPU cũng chạy được.

### Cài đặt
1. Clone hoặc tải project về máy.
2. Mở terminal, cd vào thư mục project


## Cách chạy ứng dụng web
1. Chạy server FastAPI: uvicorn app.main:app --reload

2. Mở browser truy cập:
- Trang chính (BERT pre-trained): http://127.0.0.1:8000/
- Trang mô hình tự build: http://127.0.0.1:8000/self-built

## Hướng dẫn sử dụng web

1. **Trang BERT pre-trained** (`/`)
- Nhập văn bản (ví dụ: "I love this product!" hoặc "Sản phẩm này rất tệ").
- Ấn **Dự đoán**.
- Kết quả hiển thị:
  - **Dự đoán**: Sentiment class (0 → very negative, 4 → very positive)
  - **Xác suất**: List 5 giá trị [prob class 0, 1, 2, 3, 4] – tổng ≈ 1.0
  - **Vector embedding**: 10 phần tử đầu của hidden state [CLS] token (768 chiều)

2. **Trang Self-built Transformer** (`/self-built`)
- Tương tự, nhưng dùng mô hình tự code.
- Nếu chưa train: dự đoán random (không có ý nghĩa).
- Sau khi train: dự đoán chính xác hơn (binary: Negative / Positive).

Có nút **Quay về trang BERT pre-trained** để chuyển trang dễ dàng.

## Giải thích kết quả khi ấn "Dự đoán"

### Với mô hình BERT pre-trained (`nlptown/bert-base-multilingual-uncased-sentiment`)
- **Dự đoán**: Class từ 0 đến 4
- 0: Very negative (1 sao)
- 1: Negative (2 sao)
- 2: Neutral (3 sao)
- 3: Positive (4 sao)
- 4: Very positive (5 sao)
- **Xác suất**: [p0, p1, p2, p3, p4] – giá trị cao nhất → class dự đoán
- Ví dụ: [0.001, 0.001, 0.006, 0.078, 0.914] → 91.4% very positive (class 4)
- **Vector embedding**: Đại diện ngữ nghĩa của toàn câu (10/768 chiều đầu)
- Dùng để so sánh độ tương đồng giữa các câu hoặc làm feature cho mô hình khác.

### Với mô hình Self-built (sau khi train)
- **Dự đoán**: Negative (class 0) hoặc Positive (class 1)
- **Xác suất**: [prob negative, prob positive]
- **Vector embedding**: Mean pooling của hidden states cuối (10 chiều đầu)
- Độ chính xác thường đạt ~82–88% trên test set IMDB (movie reviews).

## Cách train mô hình tự build

1. Đảm bảo đã cài thêm thư viện: 
pip install datasets tqdm scikit-learn

2. Chạy script train:
python train_self_built.py


3. Quá trình:
- Tự động tải dataset IMDB (25k train + 25k test).
- Train 5 epochs (có thể chỉnh trong code).
- Mỗi epoch in loss và accuracy trên test set.
- Thời gian: CPU ~2–5 giờ, GPU ~20–60 phút.

4. Kết thúc:
- File model được lưu: `self_built_sentiment_imdb.pt` (khoảng 30–50MB)
- Terminal in: "Đã lưu model vào self_built_sentiment_imdb.pt"

5. Sử dụng model đã train:
- Mở `app/self_built_model.py`
- Uncomment dòng:
  ```python
  self_built_model.load_state_dict(torch.load("self_built_sentiment_imdb.pt", map_location=DEVICE))

  Restart server → trang /self-built sẽ dùng model đã train.

Mẹo & Nâng cao
Muốn train nhanh hơn? 
Giảm EPOCHS = 3 hoặc BATCH_SIZE = 16 trong train_self_built.py.
Muốn 5 lớp thay vì binary? Chỉnh num_classes=5 và map label trong train script.
Muốn cải thiện accuracy? Tăng d_model=512, num_layers=4, nhead=8 (cần GPU mạnh).
Deploy: Dùng Docker hoặc Render/Heroku (cần file Procfile và requirements.txt)."# NLP_Transformer" 
