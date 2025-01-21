# Tài liệu API cho Nhận diện Giọng nói và Huấn luyện Mô hình

Tài liệu này cung cấp hướng dẫn chi tiết cách chạy API cho nhận diện giọng nói và huấn luyện mô hình, bao gồm các tham số cần thiết cho từng endpoint.

---

## **Yêu cầu**

### **Thư viện cần thiết**
Đảm bảo bạn đã cài đặt các thư viện Python sau:

- `torch`
- `fastapi`
- `uvicorn`
- `numpy`
- `jinja2`

Cài đặt thư viện bằng lệnh:
```bash
pip install -r requirements.txt
```

### **Cấu trúc thư mục**
Tạo cấu trúc thư mục sau trước khi chạy API:

```
.
├── static
├── saved_models_cross_entropy
└── data_dir
```

- `static`: Lưu trữ tạm thời các tệp tin (ví dụ: tệp đã tải lên).
- `saved_models_cross_entropy`: Thư mục lưu trữ checkpoint của mô hình.
- `data_dir`: Thư mục lưu trữ dữ liệu giọng nói của từng người.

---

## **Cách chạy API**
Khởi chạy server API bằng lệnh sau:

```bash
python app.py
```

API sẽ hoạt động tại địa chỉ `http://127.0.0.1:3000`.

---

## **Các Endpoint của API**

### **1. Xác thực Mô hình**
**Endpoint**: `/validation_model`  
**Phương thức**: POST  

#### **Tham số**:
- `test_dataset_path` (str): Đường dẫn tới tập dữ liệu kiểm tra.
- `model_name` (str): Tên mô hình cần xác thực (ví dụ: `fbanks net`).
- `use_cuda` (bool): Sử dụng GPU nếu có (`true` hoặc `false`).
- `batch_size` (int): Kích thước batch khi kiểm tra.

#### **Kết quả trả về**:
- Trả về giá trị test loss và độ chính xác trung bình.

#### **Ví dụ**:
```bash
curl -X POST http://127.0.0.1:3000/validation_model \
  -F "test_dataset_path=./data_dir/test" \
  -F "model_name=fbanks net" \
  -F "use_cuda=true" \
  -F "batch_size=32"
```

---

### **2. Huấn luyện Mô hình**
**Endpoint**: `/train_model`  
**Phương thức**: POST  

#### **Tham số**:
- `train_dataset_path` (str): Đường dẫn tới tập dữ liệu huấn luyện.
- `test_dataset_path` (str): Đường dẫn tới tập dữ liệu kiểm tra.
- `model_name` (str): Tên mô hình cần huấn luyện (ví dụ: `fbanks net`).
- `epoch` (int): Số lượng epoch.
- `lr` (float): Learning rate.
- `use_cuda` (bool): Sử dụng GPU nếu có (`true` hoặc `false`).
- `batch_size` (int): Kích thước batch khi huấn luyện và kiểm tra.

#### **Kết quả trả về**:
- Trả về các giá trị loss, độ chính xác trong quá trình huấn luyện và kiểm tra.

#### **Ví dụ**:
```bash
curl -X POST http://127.0.0.1:3000/train_model \
  -F "train_dataset_path=./data_dir/train" \
  -F "test_dataset_path=./data_dir/test" \
  -F "model_name=fbanks net" \
  -F "epoch=10" \
  -F "lr=0.001" \
  -F "use_cuda=true" \
  -F "batch_size=32"
```

---

### **3. Thêm Người Dùng**
**Endpoint**: `/add_speaker`  
**Phương thức**: POST  

#### **Tham số**:
- `file_speaker` (file): File âm thanh của người dùng.
- `speaker_name` (str): Tên của người dùng.

#### **Kết quả trả về**:
- Xác nhận file âm thanh được lưu và embeddings được tạo.

#### **Ví dụ**:
```bash
curl -X POST http://127.0.0.1:3000/add_speaker \
  -F "file_speaker=@path/to/audio.wav" \
  -F "speaker_name=JohnDoe"
```

---

### **4. Hiển thị Tất cả Người Dùng**
**Endpoint**: `/all_speaker`  
**Phương thức**: GET  

#### **Kết quả trả về**:
- Trả về danh sách tất cả người dùng đã được lưu.

#### **Ví dụ**:
```bash
curl -X GET http://127.0.0.1:3000/all_speaker
```

---

### **5. Nhận diện Giọng nói**
**Endpoint**: `/inferences`  
**Phương thức**: POST  

#### **Tham số**:
- `file_speaker` (file): File âm thanh cần nhận diện.
- `name_speaker` (str): Tên của người cần so sánh.
- `THRESHOLD` (float): Ngưỡng để xác định độ tương đồng.

#### **Kết quả trả về**:
- Thông báo liệu file âm thanh có khớp với người dùng đã lưu hay không.

#### **Ví dụ**:
```bash
curl -X POST http://127.0.0.1:3000/inferences \
  -F "file_speaker=@path/to/audio.wav" \
  -F "name_speaker=JohnDoe" \
  -F "THRESHOLD=0.7"
```

---

### **Lưu ý**
- Đảm bảo tất cả các thư mục cần thiết đã tồn tại trước khi chạy API.
- Thư mục `saved_models_cross_entropy` được sử dụng để lưu và tải checkpoint của mô hình.
- Điều chỉnh đường dẫn phù hợp với môi trường của bạn.

---




access to !link:http://localhost:3000/docs
