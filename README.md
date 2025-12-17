# Dịch máy Anh–Pháp với mô hình Encoder–Decoder LSTM

Dịch máy (Machine Translation) từ tiếng Anh sang tiếng Pháp sử dụng kiến trúc `Encoder–Decoder` với `LSTM`, huấn luyện mô hình seq2seq trên tập dữ liệu song ngữ

## Nội dung chính

- Tổng quan đề tài và kiến trúc mô hình
- Cấu trúc dữ liệu và chuẩn bị dữ liệu
- Hướng dẫn cài đặt và chạy huấn luyện (`train.py`)
- Suy luận/dự đoán với mô hình đã huấn luyện (`test.py`)
- Phân tích, đánh giá và kết quả
- Ghi chú môi trường, CUDA và xử lý sự cố

---

## 1) Kiến trúc mô hình

- **Encoder**: LSTM mã hoá chuỗi nguồn (tiếng Anh) thành véc-tơ ngữ cảnh.
- **Decoder**: LSTM giải mã véc-tơ ngữ cảnh thành chuỗi đích (tiếng Pháp), dùng cơ chế sinh token tuần tự.
- **Embedding**: Biểu diễn từ vựng ở không gian liên tục, giúp mô hình học ngữ nghĩa.
- Tuỳ chọn: có thể mở rộng với Attention (`model.py`).

Xem chi tiết hiện thực trong `model.py` và các tham số trong `config.py`.

## 2) Cấu trúc dữ liệu

Dataset - `Data/`:

```
Data/
	train/
		train.en
		train.fr
	val/
		val.en
		val.fr
	test/
		test2016/
			test_2016_flickr.en
			test_2016_flickr.fr
		test2017/
			test_2017_flickr.en
			test_2017_flickr.fr
			test_2017_mscoco.en
			test_2017_mscoco.fr
		test2018/
			test_2018_flickr.en
			test_2018_flickr.fr
```

- Mỗi file là tập câu song ngữ đã căn chỉnh dòng: `*.en` tương ứng `*.fr`.
- Tiền xử lý cơ bản (tách từ, chuẩn hoá) được thực hiện trong `data_utils.py`.

## 3) Yêu cầu môi trường

- Python 3.9+
- PyTorch (CPU hoặc CUDA)
- Các thư viện chuẩn: `numpy`, `tqdm`, ...

## 4) Cấu hình

- Chỉnh sửa tham số trong `config.py` (kích thước embedding, hidden size, số epoch, batch size, đường dẫn dữ liệu,...).
- Đường dẫn lưu mô hình tốt nhất: `models/best_model.pth`.
- Biểu đồ huấn luyện: trong thư mục `charts/`.

## 5) Huấn luyện

Chạy huấn luyện với lệnh:

```bash
python .\train.py
```

- Script sẽ đọc dữ liệu từ `Data/train` và `Data/val`, xây từ điển, tạo dataloader, và huấn luyện mô hình.
- Mô hình tốt nhất theo tiêu chí xác thực sẽ được lưu tại `models/best_model.pth`.

## 6) Suy luận/Dự đoán

Sau khi có mô hình, dịch câu mới:

```bash
python .\test.py
```

- Sửa nội dung câu nguồn cần dịch trong `test.py`.
- Script sẽ tải `models/best_model.pth` và in ra câu dịch tiếng Pháp.

## 8) Cấu trúc dự án

```
check_cuda.py     # Kiểm tra khả dụng CUDA
config.py         # Tham số huấn luyện/suy luận
data_utils.py     # Xử lý dữ liệu, từ điển, dataloader
model.py          # Định nghĩa Encoder–Decoder LSTM
test.py        	  # Suy luận: dịch câu mới
train.py          # Huấn luyện mô hình
charts/           # Biểu đồ/ảnh theo dõi huấn luyện
models/           # Lưu checkpoint/mô hình tốt nhất
```
