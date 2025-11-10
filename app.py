import streamlit as st
import torch
from PIL import Image
import pandas as pd

# Load YOLOv5 model from GitHub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Compliance mapping for 10 classes
compliance_map = {
    'Hardhat': '✅ Compliant',
    'Safety Vest': '✅ Compliant',
    'Mask': '✅ Compliant',
    'NO-Hardhat': '❌ Missing Hardhat',
    'NO-Safety Vest': '❌ Missing Vest',
    'NO-Mask': '❌ Missing Mask',
    'Person': '👤 Worker',
    'machinery': '⚙️ Machinery',
    'vehicle': '🚗 Vehicle',
    'Safety Cone': '🟠 Cone'
}

# === UPDATED UI DESIGN ===
st.set_page_config(
    page_title="SafeSite AI ⚡",
    page_icon="🦺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .stApp {
        background: transparent;
    }
    h1 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2ff, #ff00c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 style='text-align: center;'>🦺 SafeSite AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #a0e7ff;'>Real-time PPE Compliance Checker ⚡</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #e0f2ff;'>Upload a construction site image to instantly detect safety gear violations</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📤 Drop your image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("🔍 Analyzing safety compliance..."):
            results = model(image)
            detections = results.pandas().xyxy[0]

        if not detections.empty:
            # Top detection card
            top = detections.iloc[0]
            label = top['name']
            conf = round(top['confidence'] * 100, 2)
            category = compliance_map.get(label, '❓ Unknown')

            st.markdown(f"""
            <div style="background: #0e1117; padding: 20px; border-radius: 15px; border-left: 6px solid #00f2ff; box-shadow: 0 4px 15px rgba(0,242,255,0.3);">
                <h3>🎯 Primary Detection</h3>
                <p><b>Object:</b> {label}</p>
                <p><b>Confidence:</b> <span style="color:#00f2ff;font-size:1.2em">{conf}%</span></p>
                <p><b>Status:</b> {category}</p>
            </div>
            """, unsafe_allow_html=True)

            # Compliance Summary
            st.markdown("### 📊 Full Compliance Report")
            summary = detections['name'].value_counts().to_dict()
            compliant = sum(1 for k in summary.keys() if k in ['Hardhat', 'Hardhat', 'Safety Vest', 'Mask'])
            violations = sum(1 for k in summary.keys() if 'NO-' in k)
            total_workers = summary.get('Person', 0) + summary.get('NO-Hardhat', 0) + summary.get('NO-Safety Vest', 0) + summary.get('NO-Mask', 0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("👷 Workers", total_workers)
            col_b.metric("✅ Compliant", compliant, delta=f"+{compliant}")
            col_c.metric("⚠️ Violations", violations, delta=f"-{violations}" if violations > 0 else None)

            # All detections table
            st.markdown("<br>", unsafe_allow_html=True)
            styled_df = detections[['name', 'confidence']].copy()
            styled_df['confidence'] = (styled_df['confidence'] * 100).round(2)
            styled_df['Status'] = styled_df['name'].map(compliance_map).fillna('Unknown')
            styled_df = styled_df[['name', 'confidence', 'Status']]
            styled_df.rename(columns={'name': 'Detected Object', 'confidence': 'Confidence %'}, inplace=True)
            
            st.dataframe(styled_df, use_container_width=True)

            # Quick summary list
            st.markdown("### ⚡ Quick Summary")
            for cls, count in summary.items():
                emoji_label = compliance_map.get(cls, cls)
                color = "#00ff88" if "✅" in emoji_label else "#ff0066" if "❌" in emoji_label else "#ffd700"
                st.markdown(f"<span style='color:{color}; font-size:1.1em;'>• {emoji_label}: <b>{count}</b></span>", unsafe_allow_html=True)

        else:
            st.error("🚨 No objects detected. Try a clearer image of workers with PPE.")

else:
    st.info("👆 Upload an image to get started!")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #88c0ff; font-size: 13px;'>
    🚀 Powered by <b>YOLOv5</b> + <b>Streamlit</b> | Built for safer construction sites
</p>
""", unsafe_allow_html=True)
