import streamlit as st  # Thư viện để tạo giao diện Web
import pandas as pd     # Thư viện để làm việc với dữ liệu dạng bảng (DataFrame)
import numpy as np      # Thư viện toán học, làm việc với mảng
import joblib           # Thư viện dùng để tải (load) mô hình AI đã được lưu (.pkl)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Công cụ chuẩn hóa dữ liệu và mã hóa nhãn
from sklearn.model_selection import train_test_split  # Hàm dùng để chia tập dữ liệu

# ==========================================
# 1. TỪ ĐIỂN DỊCH TÊN CÂY TRỒNG
# ==========================================
# Mô hình dự đoán trả về tên cây bằng tiếng Anh (do data gốc là tiếng Anh).
# Từ điển này giúp tự động dịch tên cây đó sang tiếng Việt để giao diện thân thiện hơn.
crop_dict = {
    'apple': 'Táo',
    'banana': 'Chuối',
    'blackgram': 'Đậu đen',
    'chickpea': 'Đậu gà',
    'coconut': 'Dừa',
    'coffee': 'Cà phê',
    'cotton': 'Bông',
    'grapes': 'Nho',
    'jute': 'Đay',
    'kidneybeans': 'Đậu tây',
    'lentil': 'Đậu lăng',
    'maize': 'Ngô (Bắp)',
    'mango': 'Xoài',
    'mothbeans': 'Đậu moth',
    'mungbean': 'Đậu xanh',
    'muskmelon': 'Dưa lưới',
    'orange': 'Cam',
    'papaya': 'Đu đủ',
    'pigeonpeas': 'Đậu cajan',
    'pomegranate': 'Lựu',
    'rice': 'Lúa gạo',
    'watermelon': 'Dưa hấu'
}

# ==========================================
# 2. CẤU HÌNH GIAO DIỆN WEB
# ==========================================
# Thiết lập tiêu đề hiển thị trên tab trình duyệt và icon
st.set_page_config(page_title="Hệ thống Gợi ý Cây trồng", page_icon="🌱", layout="centered")

# Tiêu đề chính hiển thị trên trang web
st.title("🌱 HỆ THỐNG GỢI Ý CÂY TRỒNG (Crop Recommendation)")
# Lời chào / Hướng dẫn người dùng
st.write("Vui lòng nhập các thông số của đất và môi trường để nhận gợi ý loại cây trồng phù hợp nhất!")

# ==========================================
# 3. HÀM TẢI MÔ HÌNH VÀ TÁI TẠO TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
# @st.cache_resource giúp Streamlit lưu lại kết quả của hàm này vào bộ nhớ tạm (Cache).
# Việc này đảm bảo khi người dùng thao tác trên web, máy không phải load lại file .pkl và tính toán lại từ đầu, giúp web chạy cực nhanh.
@st.cache_resource
def load_and_prep_model():
    # 3.1. Tải mô hình Random Forest tốt nhất đã huấn luyện từ trước
    model = joblib.load('models/random_forest_model.pkl')
    
    # 3.2. Đọc lại file dữ liệu gốc để thiết lập lại các công cụ xử lý
    # Mục đích: Dữ liệu người dùng nhập vào Web là số thật, nhưng mô hình chỉ nhận số Đã Chuẩn Hóa.
    # Nên ta phải tái tạo lại cái khuôn chuẩn hóa hệt như lúc huấn luyện mô hình.
    raw_data = pd.read_csv('data/raw/Crop_recommendation.csv')
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    
    # Tách dữ liệu thành 2 phần: X (Đặc trưng) và y (Nhãn - Tên cây)
    X = raw_data[features].copy()
    y = raw_data['label'].copy()
    
    # 3.3. Mã hóa biến mục tiêu y từ dạng chữ (chuỗi) sang số (0-21)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 3.4. Chia dữ liệu thành tập Train (80%) và Test (20%) y hệt như file 02_preprocessing.ipynb
    # Việc chia tách phải sử dụng 'random_state=42' để đảm bảo máy cắt đúng cái khúc dữ liệu cũ.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 3.5. Xử lý giá trị ngoại lệ (Outlier Capping) bằng phương pháp IQR (Interquartile Range)
    # Thuật toán sẽ tính toán ra cái 'Ngưỡng dưới' (lower_bound) và 'Ngưỡng trên' (upper_bound) dựa trên tập Train.
    lower_bounds = {}
    upper_bounds = {}
    X_train_clean = X_train.copy()
    
    for feature in features:
        # Q1 là mốc 25%, Q3 là mốc 75%
        Q1 = X_train_clean[feature].quantile(0.25)
        Q3 = X_train_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Công thức chuẩn của IQR
        lower_bounds[feature] = Q1 - 1.5 * IQR
        upper_bounds[feature] = Q3 + 1.5 * IQR
        
        # Dùng hàm clip để 'ép' (chặn) các giá trị vượt quá ngưỡng trở về đúng ngưỡng.
        X_train_clean[feature] = X_train_clean[feature].clip(lower=lower_bounds[feature], upper=upper_bounds[feature])
        
    # 3.6. Chuẩn hóa dữ liệu (Standard Scaler)
    # Scale dữ liệu giúp đồng bộ thang đo của tất cả các cột. Hàm fit() sẽ học thuộc phương sai và độ lệch chuẩn của tập Train.
    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    
    # Trả về các công cụ cần thiết ra ngoài để tái sử dụng khi người dùng bấm dự đoán
    return model, scaler, lower_bounds, upper_bounds, label_encoder

# Kích hoạt hàm bên trên và lấy ra 5 công cụ
model, scaler, lower_bounds, upper_bounds, label_encoder = load_and_prep_model()

# ==========================================
# 4. GIAO DIỆN NHẬP LIỆU (FORM)
# ==========================================
# st.form giúp gom nhóm các thanh trượt lại. 
# Khi nào người dùng bấm nút "Submit" thì web mới load 1 lần, tránh việc vừa kéo thanh trượt web đã tính toán loạn lên.
with st.form("my_form"):
    # Chia giao diện làm 2 cột cho gọn và đẹp mắt
    col1, col2 = st.columns(2)
    
    with col1:
        # st.slider: Tạo thanh trượt chọn số. Nhận vào (Tên hiển thị, Nhỏ nhất, Lớn nhất, Giá trị mặc định, Bước nhảy)
        N = st.slider("Nitơ (N) trong đất", min_value=0.0, max_value=140.0, value=90.0, step=1.0)
        P = st.slider("Photpho (P) trong đất", min_value=5.0, max_value=145.0, value=42.0, step=1.0)
        K = st.slider("Kali (K) trong đất", min_value=5.0, max_value=205.0, value=43.0, step=1.0)
        temp = st.slider("Nhiệt độ (°C)", min_value=8.0, max_value=45.0, value=20.8, step=0.1)
        
    with col2:
        hum = st.slider("Độ ẩm (%)", min_value=14.0, max_value=100.0, value=82.0, step=0.1)
        ph = st.slider("Độ pH của đất", min_value=3.5, max_value=10.0, value=6.5, step=0.1)
        rain = st.slider("Lượng mưa (mm)", min_value=20.0, max_value=300.0, value=202.9, step=0.1)
        
    # Nút bấm để xác nhận
    submitted = st.form_submit_button("🚀 Đưa ra gợi ý!")

# ==========================================
# 5. XỬ LÝ DỮ LIỆU ĐẦU VÀO & XUẤT DỰ ĐOÁN
# ==========================================
# Khối lệnh này chỉ chạy KHI VÀ CHỈ KHI người dùng bấm cái nút "Đưa ra gợi ý!" ở trên
if submitted:
    # 5.1. Bọc 7 thông số người dùng vừa kéo thành một bộ dữ liệu bảng (DataFrame) 1 dòng
    input_data = pd.DataFrame({
        "N": [N], "P": [P], "K": [K], 
        "temperature": [temp], "humidity": [hum], 
        "ph": [ph], "rainfall": [rain]
    })
    
    # 5.2. Chặn ngoại lệ (Capping)
    # Nếu người dùng cố tình kéo một chỉ số cao bất thường (VD: Lượng mưa = 300), 
    # hàm clip() sẽ ép nó về đúng cái ngưỡng upper_bound mà lúc nãy ta đã học từ tập Train.
    for feature in input_data.columns:
        input_data[feature] = input_data[feature].clip(
            lower=lower_bounds[feature], 
            upper=upper_bounds[feature]
        )
        
    # 5.3. Chuẩn hóa (Scaling)
    # Đưa bộ số vừa nhận về dạng âm/dương (VD: 0.1, -1.2, 0.5...) hệt như cách tập Train đã bị biến đổi
    input_scaled = scaler.transform(input_data)
    
    # 5.4. Trí tuệ nhân tạo (Model) làm việc
    # Đưa bộ số đã chuẩn hóa vào cho Random Forest phán đoán. Nó sẽ trả ra 1 con số từ 0 -> 21
    pred = model.predict(input_scaled)
    
    # 5.5. Giải mã từ số (0-21) về lại thành Tên Tiếng Anh (VD: 20 -> 'rice')
    pred_label_eng = label_encoder.inverse_transform(pred)[0]
    
    # 5.6. Dùng từ điển tra cứu để dịch tên tiếng Anh đó sang Tiếng Việt
    pred_label_vi = crop_dict.get(pred_label_eng, pred_label_eng) # Nếu không có trong từ điển thì in luôn tiếng Anh
    
    # 5.7. In kết quả đẹp mắt ra màn hình
    # st.success() sẽ tạo một cái khung viền màu xanh lá cây tượng trưng cho thành công.
    st.success(f"### 🎉 Cây trồng phù hợp nhất: **{pred_label_vi}**")
    # st.info() tạo viền màu xanh dương để in ra thông tin phụ (tên gốc)
    st.info(f"*(Tên gốc tiếng Anh: {pred_label_eng})*")
