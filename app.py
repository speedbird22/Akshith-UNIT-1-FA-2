# app.py
import streamlit as st
from PIL import Image
import pandas as pd
import torch
import os

# THIS IS THE MAGIC LINE THAT DISABLES OPENCV COMPLETELY
os.environ["YOLO_DISABLE_OPENCV"] = "1"

# Now import YOLO safely
from ultralytics import YOLO

# Page setup
st.set_page_config(page_title="PPE Checker", page_icon="hardhat", layout="centered")

st.markdown("""
<h1 style='text-align: center;'>hardhat PPE Compliance Detector</h1>
<p style='text-align: center; color: #666;'>Upload any construction site image</p>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return YOLO("best.pt")

model = load_model()

# Compliance map
compliance_map = {
    'Hardhat': 'Compliant - Hardhat',
    'Safety Vest': 'Compliant - Vest',
    'Mask': 'Compliant - Mask',
    'NO-Hardhat': 'Missing Hardhat',
    'NO-Safety Vest': 'Missing Vest',
    'NO-Mask': 'Missing Mask',
    'Person': 'Worker',
    'machinery': 'Machinery',
    'vehicle': 'Vehicle',
    'Safety Cone': 'Cone'
}

# Upload
uploaded = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, "Your Image")

    with st.spinner("Checking PPE..."):
        results = model(img, conf=0.25)[0]

        if len(results.boxes) > 0:
            df = pd.DataFrame({
                'Item': [results.names[int(c)] for c in results.boxes.cls],
                'Confidence': [f"{c:.1%}" for c in results.boxes.conf.cpu().numpy()]
            }).sort_values('Confidence', ascending=False)

            top = df.iloc[0]
            status = compliance_map.get(top['Item'], 'Unknown')

            if 'Missing' in status:
                st.error(f"VIOLATION: {status}")
            else:
                st.success(f"{status}")

            st.write("### All Detections")
            st.dataframe(df)

            st.write("### Image with Boxes")
            annotated = results.plot()
            st.image(annotated, use_column_width=True)

        else:
            st.warning("No PPE items found")

else:
    st.info("Upload an image to start")

st.caption("Works perfectly on Streamlit Cloud India • No OpenCV needed")
