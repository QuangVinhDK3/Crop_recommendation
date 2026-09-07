# 🌾 CROP RECOMMENDATION PROJECT (Gợi Ý Cây Trồng)

**Đồ án Machine Learning (BT09):** Xây dựng hệ thống gợi ý loại cây trồng phù hợp dựa trên các đặc tính hóa học của đất và điều kiện khí hậu. 

## 📖 Mô tả chi tiết đồ án
Nông nghiệp đóng vai trò rất quan trọng đối với nền kinh tế. Việc lựa chọn sai loại cây trồng so với điều kiện thổ nhưỡng có thể dẫn đến rủi ro lớn về năng suất. Đồ án này áp dụng **Machine Learning** để giải quyết bài toán đó bằng cách phân tích dữ liệu đất đai và thời tiết, từ đó đưa ra quyết định canh tác tối ưu nhất.

- **Tập dữ liệu (Dataset):** Dữ liệu thu thập từ Kaggle, bao gồm 2200 mẫu phân bổ đều cho 22 loại cây trồng khác nhau (Lúa, Cà phê, Táo, Xoài...).
- **Đặc trưng (Features):** Hệ thống dựa trên 7 yếu tố đầu vào bao gồm hàm lượng Nito (`N`), Photpho (`P`), Kali (`K`), Nhiệt độ (`temperature`), Độ ẩm (`humidity`), Độ pH (`ph`), và Lượng mưa (`rainfall`).
- **Mục tiêu (Target):** Phân loại và dự đoán chính xác Nhãn (`label`) - tên loại cây trồng nên được canh tác.
- **Quy trình thực hiện (Pipeline):**
  1. **EDA & Visualization:** Phân tích dữ liệu, vẽ biểu đồ phân bố và ma trận tương quan giữa độ ẩm, lượng mưa và nhiệt độ.
  2. **Preprocessing:** Làm sạch dữ liệu, chuẩn hóa tỉ lệ (`StandardScaler`) và mã hóa tên cây trồng (`LabelEncoder`).
  3. **Clustering (Gom cụm):** Ứng dụng thuật toán **K-Means** và **Agglomerative Clustering** để tìm kiếm các cụm đất có đặc tính tương đồng, đánh giá qua Silhouette Score.
  4. **Classification (Phân lớp):** Áp dụng **Random Forest**, **Support Vector Machine (SVM)** và **Decision Tree**. Tối ưu thông số (GridSearchCV) và đánh giá độ tin cậy bằng các độ đo Accuracy, Precision, Recall cùng Confusion Matrix.

---

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

### Bước 5: Trải nghiệm giao diện Web App (Demo)
Dự án có đi kèm một giao diện Web trực quan để tương tác với mô hình dự đoán. Để chạy trang web, bạn mở Terminal và gõ lệnh:
```bash
python -m streamlit run app.py
```
Trang web sẽ tự động mở lên trên trình duyệt (thường ở địa chỉ `http://localhost:8501`). Tại đây, bạn có thể tự tay kéo thanh trượt thông số N, P, K... và xem AI gợi ý loại cây trồng phù hợp nhất bằng tiếng Việt.
---

## 3. Kết quả huấn luyện mô hình (Model Results)

Trong quá trình thực nghiệm, nhóm đã thử nghiệm và đánh giá ba mô hình phân lớp chính:
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Decision Tree**

**Kết quả đánh giá trên tập Test:**
- **Random Forest** thể hiện hiệu năng xuất sắc nhất với độ chính xác tổng thể (Accuracy) lên tới **>99%**, cũng như các chỉ số Precision và Recall đều đạt mức tương đương.
- **Decision Tree** có tốc độ huấn luyện vô cùng nhanh và đóng vai trò tốt trong việc làm mô hình cơ sở.
- Mô hình tốt nhất (Random Forest) đã được lưu thành công dưới dạng file `.pkl` trong thư mục `models/` và sẵn sàng để tích hợp hoặc sử dụng để dự đoán dữ liệu mới.
