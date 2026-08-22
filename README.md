# Crop Recommendation Project

Đồ án học máy: Gợi ý loại cây trồng phù hợp dựa trên đặc tính đất và khí hậu (Crop Recommendation Dataset).

## 1. Cấu trúc thư mục

Dự án được cấu trúc như sau:

- `data/raw/`: Chứa dữ liệu gốc (`Crop_recommendation.csv`).
- `data/processed/`: Chứa dữ liệu sau khi tiền xử lý (nếu có).
- `notebooks/`: Chứa các file Jupyter Notebook (`.ipynb`) dùng để làm bài: EDA, Tiền xử lý, Phân lớp (Classification) và Gom cụm (Clustering).
- `models/`: Chứa các file mô hình đã được huấn luyện.
- `reports/figures/`: Nơi xuất và lưu các biểu đồ trực quan hóa.
- `src/`: (Tùy chọn) Chứa mã nguồn Python tái sử dụng.

---

## 2. Hướng dẫn cài đặt và chạy dự án (Dành cho thành viên nhóm)

Để chạy được đồ án này trên máy cá nhân sau khi clone về, các bạn hãy làm theo các bước sau:

### Bước 1: Clone dự án về máy
Mở Terminal / Command Prompt và chạy lệnh:
```bash
git clone <URL_CỦA_REPO>
cd Crop_recommendation
```

### Bước 2: Kiểm tra phiên bản Python
Đảm bảo máy bạn đã cài đặt Python (khuyên dùng Python 3.8 trở lên). Kiểm tra bằng lệnh:
```bash
python --version
```
*(Nếu chưa có, vui lòng lên trang chủ python.org để tải và cài đặt, nhớ tích chọn "Add Python to PATH")*

### Bước 3: Cài đặt các thư viện cần thiết
Dự án yêu cầu các thư viện cốt lõi như pandas, scikit-learn, matplotlib, seaborn và công cụ Jupyter. Bạn cài đặt toàn bộ bằng lệnh sau:
```bash
pip install -r requirements.txt
```
*(Lưu ý: Nếu lệnh pip bị lỗi, hãy thử dùng `python -m pip install -r requirements.txt`)*

### Bước 4: Khởi chạy và lập trình
- **Nếu bạn dùng Visual Studio Code (Khuyên dùng)**: Mở thư mục dự án bằng VS Code, sau đó cài đặt extension "Jupyter" và "Python". Mở các file trong thư mục `notebooks/` và có thể chạy trực tiếp từng cell code.
- **Nếu bạn dùng trình duyệt web**: Chạy lệnh sau trong Terminal để mở Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
  Sau đó điều hướng vào thư mục `notebooks/` để mở các file `.ipynb`.
