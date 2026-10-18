import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 1. TỪ ĐIỂN VÀ DỮ LIỆU TĨNH
# ==========================================
crop_dict = {
    'apple':       ('Táo',        '🍎'),
    'banana':      ('Chuối',      '🍌'),
    'blackgram':   ('Đậu đen',    '🫘'),
    'chickpea':    ('Đậu gà',     '🫘'),
    'coconut':     ('Dừa',        '🥥'),
    'coffee':      ('Cà phê',     '☕'),
    'cotton':      ('Bông',       '🌿'),
    'grapes':      ('Nho',        '🍇'),
    'jute':        ('Đay',        '🌿'),
    'kidneybeans': ('Đậu tây',    '🫘'),
    'lentil':      ('Đậu lăng',   '🫘'),
    'maize':       ('Ngô (Bắp)',  '🌽'),
    'mango':       ('Xoài',       '🥭'),
    'mothbeans':   ('Đậu moth',   '🫘'),
    'mungbean':    ('Đậu xanh',   '🫘'),
    'muskmelon':   ('Dưa lưới',   '🍈'),
    'orange':      ('Cam',        '🍊'),
    'papaya':      ('Đu đủ',      '🍈'),
    'pigeonpeas':  ('Đậu cajan',  '🫘'),
    'pomegranate': ('Lựu',        '🍎'),
    'rice':        ('Lúa gạo',    '🌾'),
    'watermelon':  ('Dưa hấu',    '🍉'),
}

# Feature Importance từ kết quả Random Forest (notebook 03)
FEATURE_IMPORTANCE = {
    'rainfall':    0.2385,
    'humidity':    0.2161,
    'K':           0.1781,
    'P':           0.1529,
    'N':           0.0986,
    'temperature': 0.0664,
    'ph':          0.0494,
}

FEATURE_LABEL = {
    'rainfall':    '🌧️ Lượng mưa',
    'humidity':    '💧 Độ ẩm',
    'K':           '🧪 Kali (K)',
    'P':           '🧪 Photpho (P)',
    'N':           '🧪 Nitơ (N)',
    'temperature': '🌡️ Nhiệt độ',
    'ph':          '⚗️ pH đất',
}

# ==========================================
# 2. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Hệ thống Gợi ý Cây trồng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Header chính */
    .main-header {
        text-align: center;
        padding: 1.4rem 1rem;
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 15px rgba(45,106,79,0.35);
    }
    .main-header h1 { margin: 0; font-size: 2rem; letter-spacing: 1px; }
    .main-header p  { margin: 0.4rem 0 0; font-size: 0.95rem; opacity: 0.88; }

    /* Hộp kết quả */
    .result-box {
        text-align: center;
        padding: 1.8rem 1rem;
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        border-radius: 14px;
        color: white;
        box-shadow: 0 4px 20px rgba(45,106,79,0.4);
    }
    .result-crop  { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
    .result-en    { font-size: 0.95rem; opacity: 0.78; margin-bottom: 0.8rem; }
    .result-conf  { font-size: 1.15rem; font-weight: 600; }

    /* Top-3 card */
    .top3-card {
        background: #f0faf4;
        border-left: 4px solid #52b788;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.55rem;
    }
    .top3-name   { font-weight: 600; font-size: 1rem; color: #1b4332; }
    .top3-en     { font-size: 0.8rem; color: #6c757d; }
    .top3-prob   { float: right; font-weight: 700; color: #2d6a4f; font-size: 1rem; }

    /* Section tiêu đề */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b4332;
        border-bottom: 2px solid #52b788;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Metric card */
    div[data-testid="metric-container"] {
        background: #f0faf4;
        border: 1px solid #b7e4c7;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }

    /* Nút */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        transition: transform 0.15s, box-shadow 0.15s;
        box-shadow: 0 3px 10px rgba(45,106,79,0.3);
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(45,106,79,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🌱 HỆ THỐNG GỢI Ý CÂY TRỒNG</h1>
    <p>Nhập thông số đất và khí hậu để nhận gợi ý loại cây trồng phù hợp nhất · Mô hình Random Forest · Accuracy 99.55%</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 📊 Thông tin Mô hình")
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy",  "99.55%")
    col_a.metric("Recall",    "99.55%")
    col_b.metric("Precision", "99.57%")
    col_b.metric("Số lớp",   "22")

    st.markdown("---")
    st.markdown("## 🏆 Độ quan trọng đặc trưng")
    st.caption("Mức độ ảnh hưởng của từng yếu tố đến kết quả dự đoán")

    sorted_fi = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1])
    fi_labels = [FEATURE_LABEL[k] for k, _ in sorted_fi]
    fi_values = [v for _, v in sorted_fi]
    fi_colors = [
        '#1b4332' if v >= 0.20 else
        '#2d6a4f' if v >= 0.15 else
        '#40916c' if v >= 0.10 else
        '#52b788' if v >= 0.07 else
        '#74c69d'
        for v in fi_values
    ]

    fig_fi = go.Figure(go.Bar(
        x=fi_values,
        y=fi_labels,
        orientation='h',
        marker_color=fi_colors,
        text=[f"{v:.1%}" for v in fi_values],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig_fi.update_layout(
        margin=dict(l=0, r=40, t=5, b=5),
        height=260,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, max(fi_values) * 1.35]),
        yaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("## 📐 Phạm vi giá trị")
    st.dataframe(
        pd.DataFrame({
            'Đặc trưng':  ['N', 'P', 'K', 'Nhiệt độ', 'Độ ẩm', 'pH', 'Lượng mưa'],
            'Min':        [0,   5,   5,   8.8,        14.3,    3.5,  20.2],
            'TB':         [50.6, 53.4, 48.1, 25.6,   71.5,    6.5,  103.5],
            'Max':        [140, 145, 205,  43.7,       100.0,   9.9,  298.6],
        }),
        hide_index=True,
        use_container_width=True,
    )

# ==========================================
# 5. TẢI MÔ HÌNH
# ==========================================
@st.cache_resource
def load_and_prep_model():
    """Tải mô hình và tái tạo pipeline tiền xử lý đúng như notebook 02."""
    model = joblib.load('models/random_forest_model.pkl')

    raw_data = pd.read_csv('data/raw/Crop_recommendation.csv')
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    X = raw_data[features].copy()
    y = raw_data['label'].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Chia Train/Test đúng như notebook 02
    X_train, _, y_train, _ = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # IQR Capping — chỉ học từ X_train (tránh data leakage)
    lower_bounds, upper_bounds = {}, {}
    X_train_clean = X_train.copy()
    for feature in features:
        Q1 = X_train_clean[feature].quantile(0.25)
        Q3 = X_train_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds[feature] = Q1 - 1.5 * IQR
        upper_bounds[feature] = Q3 + 1.5 * IQR
        X_train_clean[feature] = X_train_clean[feature].clip(
            lower=lower_bounds[feature], upper=upper_bounds[feature]
        )

    # StandardScaler — chỉ fit trên X_train
    scaler = StandardScaler()
    scaler.fit(X_train_clean)

    return model, scaler, lower_bounds, upper_bounds, label_encoder, features

model, scaler, lower_bounds, upper_bounds, label_encoder, features = load_and_prep_model()

# ==========================================
# 6. NHẬP LIỆU
# ==========================================
st.markdown('<div class="section-title">🔬 Nhập thông số đất và môi trường</div>', unsafe_allow_html=True)

col_soil, col_climate = st.columns(2, gap="large")

with col_soil:
    st.markdown("**🌍 Dinh dưỡng đất**")
    N = st.slider(
        "🧪 Nitơ (N) — mg/kg",
        min_value=0.0, max_value=140.0, value=50.0, step=1.0,
        help="Hàm lượng Nitơ trong đất · Nitơ thúc đẩy sinh trưởng lá và thân · Mức trung bình: 50.6 mg/kg",
    )
    P = st.slider(
        "🧪 Photpho (P) — mg/kg",
        min_value=5.0, max_value=145.0, value=53.0, step=1.0,
        help="Hàm lượng Photpho trong đất · Hỗ trợ phát triển rễ và hoa · Mức trung bình: 53.4 mg/kg",
    )
    K = st.slider(
        "🧪 Kali (K) — mg/kg  ⭐ quan trọng thứ 3",
        min_value=5.0, max_value=205.0, value=48.0, step=1.0,
        help="Hàm lượng Kali trong đất · Giúp cây kháng bệnh & phát triển quả · Mức trung bình: 48.1 mg/kg",
    )
    ph = st.slider(
        "⚗️ Độ pH của đất",
        min_value=3.5, max_value=10.0, value=6.5, step=0.1,
        help="pH lý tưởng cho đa số cây: 5.5–7.0 · Mức trung bình: 6.5",
    )

with col_climate:
    st.markdown("**🌤️ Điều kiện khí hậu**")
    rain = st.slider(
        "🌧️ Lượng mưa (mm)  ⭐ quan trọng nhất",
        min_value=20.0, max_value=300.0, value=103.0, step=0.5,
        help="Tổng lượng mưa · Yếu tố ảnh hưởng lớn nhất (23.85%) · Mức trung bình: 103.5 mm",
    )
    hum = st.slider(
        "💧 Độ ẩm (%)  ⭐ quan trọng thứ 2",
        min_value=14.0, max_value=100.0, value=71.0, step=0.5,
        help="Độ ẩm không khí · Yếu tố quan trọng thứ 2 (21.61%) · Mức trung bình: 71.5%",
    )
    temp = st.slider(
        "🌡️ Nhiệt độ (°C)",
        min_value=8.0, max_value=45.0, value=25.6, step=0.1,
        help="Nhiệt độ môi trường · Mức trung bình: 25.6 °C",
    )

# ==========================================
# 7. NÚT DỰ ĐOÁN
# ==========================================
st.markdown("---")
submitted = st.button("🚀 Đưa ra gợi ý cây trồng!", use_container_width=True, type="primary")

# ==========================================
# 8. KẾT QUẢ DỰ ĐOÁN
# ==========================================
if submitted:
    input_data = pd.DataFrame({
        "N": [N], "P": [P], "K": [K],
        "temperature": [temp], "humidity": [hum],
        "ph": [ph], "rainfall": [rain],
    })

    # IQR Capping
    for feature in input_data.columns:
        input_data[feature] = input_data[feature].clip(
            lower=lower_bounds[feature],
            upper=upper_bounds[feature],
        )

    # Scale & Dự đoán
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    pred_proba = model.predict_proba(input_scaled)[0]

    # Giải mã nhãn
    pred_label_eng = label_encoder.inverse_transform(pred)[0]
    vi_name, emoji = crop_dict.get(pred_label_eng, (pred_label_eng, '🌿'))

    # Top-3
    top3_idx   = np.argsort(pred_proba)[::-1][:3]
    top3_eng   = label_encoder.inverse_transform(top3_idx)
    top3_proba = pred_proba[top3_idx]

    confidence = top3_proba[0] * 100

    st.markdown("---")
    st.markdown("## 🎯 Kết quả dự đoán")

    col_res, col_top3 = st.columns([1, 1], gap="large")

    # ── Kết quả chính ──
    with col_res:
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:3.5rem">{emoji}</div>
            <div class="result-crop">{vi_name}</div>
            <div class="result-en">({pred_label_eng})</div>
            <div class="result-conf">Độ tin cậy: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': '%', 'font': {'size': 28, 'color': '#1b4332'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#555'},
                'bar': {'color': '#2d6a4f'},
                'steps': [
                    {'range': [0,  50], 'color': '#d8f3dc'},
                    {'range': [50, 80], 'color': '#95d5b2'},
                    {'range': [80, 100], 'color': '#52b788'},
                ],
                'threshold': {
                    'line': {'color': '#1b4332', 'width': 3},
                    'thickness': 0.8,
                    'value': confidence,
                },
            },
            domain={'x': [0, 1], 'y': [0, 1]},
        ))
        fig_gauge.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Top-3 ──
    with col_top3:
        st.markdown("### 🏅 Top 3 gợi ý")
        medals = ["🥇", "🥈", "🥉"]
        for i, (eng, prob) in enumerate(zip(top3_eng, top3_proba)):
            vi, em = crop_dict.get(eng, (eng, '🌿'))
            st.markdown(f"""
            <div class="top3-card">
                <span class="top3-prob">{prob:.1%}</span>
                <div class="top3-name">{medals[i]} {em} {vi}</div>
                <div class="top3-en">{eng}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

    # ── Thông số đã nhập ──
    st.markdown("---")
    st.markdown("### 📋 Thông số bạn đã nhập")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧪 Nitơ (N)",    f"{N:.0f} mg/kg")
    c2.metric("🧪 Photpho (P)", f"{P:.0f} mg/kg")
    c3.metric("🧪 Kali (K)",    f"{K:.0f} mg/kg")
    c4.metric("⚗️ pH",          f"{ph:.1f}")

    c5, c6, c7, _ = st.columns(4)
    c5.metric("🌡️ Nhiệt độ",  f"{temp:.1f} °C")
    c6.metric("💧 Độ ẩm",     f"{hum:.1f} %")
    c7.metric("🌧️ Lượng mưa", f"{rain:.1f} mm")
