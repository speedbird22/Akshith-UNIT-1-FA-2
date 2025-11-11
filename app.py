import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile

# ============================
# Page Config & Custom CSS
# ============================
st.set_page_config(
    page_title="PPE Compliance Detector",
    page_icon="🦺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(to bottom right, #f0f2f6, #e0e7ff);
    }
    .header-title {
        font-size: 3rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 3px dashed #3b82f6;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fafc;
        transition: all 0.3s;
    }
    .upload-box:hover {
        border-color: #1e40af;
        background-color: #eef2ff;
    }
    .summary-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .compliant { color: #10b981; font-weight: bold; }
    .non-compliant { color: #ef4444; font-weight: bold; }
    .neutral { color: #6366f1; font-weight: bold; }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()

# ============================
# Compliance Map (with emojis)
# ============================
compliance_map = {
    'Hardhat': '✅ Hardhat Worn',
    'Safety Vest': '✅ Safety Vest Worn',
    'Mask': '✅ Mask Worn',
    'NO-Hardhat': '❌ Missing Hardhat',
    'NO-Safety Vest': '❌ Missing Safety Vest',
    'NO-Mask': '❌ Missing Mask',
    'Person': '👤 Worker Detected',
    'machinery': '⚙️ Machinery',
    'vehicle': '🚗 Vehicle',
    'Safety Cone': '🟠 Safety Cone'
}

# ============================
# UI Header
# ============================
st.markdown('<h1 class="header-title">🦺 PPE Compliance Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Upload a construction site image to instantly detect workers and verify PPE compliance.</p>', unsafe_allow_html=True)

# ============================
# File Uploader with Custom Box
# ============================
st.markdown("""
<div class="upload-box">
    <h3>📤 Upload Construction Site Image</h3>
    <p>Supported formats: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)
    
    with st.spinner("🔍 Analyzing image with YOLOv5..."):
        results = model(image)
        df = results.pandas().xyxy[0]
        
        # Annotate image
        annotated_img = np.array(image)
        for _, row in df.iterrows():
            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            color = (0, 255, 0) if 'NO-' not in label and label in ['Hardhat', 'Safety Vest', 'Mask'] else (255, 0, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            display_text = compliance_map.get(label, label)
            cv2.putText(annotated_img, display_text, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    with col2:
        st.markdown("### 🧠 Detection Result")
        st.image(annotated_img, use_column_width=True)

    # ============================
    # Compliance Summary Card
    # ============================
    st.markdown("### 📋 Compliance Summary")
    st.markdown("<div class='summary-card'>", unsafe_allow_html=True)

    compliant_count = 0
    violations = 0
    workers = df[df['name'] == 'Person'].shape[0]

    for label in df['name'].unique():
        count = (df['name'] == label).sum()
        text = compliance_map.get(label, label)
        if '✅' in text:
            st.markdown(f"<p class='compliant'>✅ {text}: <strong>{count}</strong></p>", unsafe_allow_html=True)
            compliant_count += count
        elif '❌' in text:
            st.markdown(f"<p class='non-compliant'>❌ {text}: <strong>{count}</strong></p>", unsafe_allow_html=True)
            violations += count
        else:
            st.markdown(f"<p class='neutral'>{text}: <strong>{count}</strong></p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # Overall Status
    # ============================
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        st.metric("👷 Total Workers", workers)
    with col_status2:
        st.metric("✅ Compliant Items", compliant_count)
    with col_status3:
        st.metric("⚠️ Violations", violations, delta=f"-{violations}" if violations == 0 else None)

    if violations == 0 and workers > 0:
        st.success("🎉 **All workers are fully PPE compliant!** Great job maintaining safety standards!")
    elif violations > 0:
        st.error(f"🚨 **{violations} PPE violation(s) detected!** Immediate action required.")

else:
    st.info("👆 Please upload an image to begin detection.")

# ============================
# Footer
# ============================
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using <strong>YOLOv5</strong> + <strong>Streamlit</strong> | Powered by Ultralytics</p>
    <p>Ensuring safety, one detection at a time.</p>
</div>
""", unsafe_allow_html=True)
