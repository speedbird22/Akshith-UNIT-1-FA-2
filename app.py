import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Load YOLO model (make sure best.pt is in the same directory or provide full path)
model = YOLO('best.pt')

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

# Streamlit UI setup
st.set_page_config(page_title="Construction PPE Dashboard", page_icon="👷", layout="centered")
st.markdown("<h1 style='text-align: center;'>👷 Construction Site PPE Compliance</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to detect workers and assess safety compliance.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Run inference
    results = model.predict(image, save=False, conf=0.25)
    detections = results[0].boxes

    if detections is not None and detections.cls.numel() > 0:
        names = results[0].names
        df = detections.pandas().xyxy[0]

        top = df.iloc[0]
        label = top['name']
        conf = round(top['confidence'] * 100, 2)
        category = compliance_map.get(label, 'Unknown')

        st.markdown("### 🧾 Top Detection")
        st.success(f"**Detected:** {label}")
        st.info(f"**Confidence:** {conf}%")
        if category.startswith("✅"):
            st.success(f"**Compliance:** {category}")
        elif category.startswith("❌"):
            st.warning(f"**Violation:** {category}")
        else:
            st.info(f"**Category:** {category}")

        # Show full detection table
        st.markdown("### 📋 All Detections")
        st.dataframe(df[['name', 'confidence', 'class']])

        # Compliance summary
        st.markdown("### 📊 Compliance Summary")
        summary = df['name'].value_counts().to_dict()
        for cls, count in summary.items():
            label = compliance_map.get(cls, cls)
            st.write(f"- {label}: {count}")
    else:
        st.error("🚫 No PPE-related objects detected. Try another image.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using YOLO and Streamlit</p>", unsafe_allow_html=True)
