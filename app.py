# app.py
import streamlit as st
from PIL import Image
import pandas as pd
import torch
import os

# FIX 1: Force cv2 to load before ultralytics
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2

# FIX 2: Import YOLO AFTER cv2
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="PPE Compliance Detector",
    page_icon="hardhat",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("""
<h1 style='text-align: center;'>hardhat Construction PPE Compliance Checker</h1>
<p style='text-align: center;'>Upload image → Instant safety audit</p>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading YOLOv5 model (first time takes 15-20 sec)...")
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Compliance mapping
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

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting PPE..."):
        results = model(image, conf=0.25, imgsz=640)[0]

        if len(results.boxes) > 0:
            names = results.names
            df = pd.DataFrame({
                'Object': [names[int(c)] for c in results.boxes.cls],
                'Confidence': [f"{c:.1%}" for c in results.boxes.conf.tolist()]
            }).sort_values(by='Confidence', ascending=False).reset_index(drop=True)

            # Top detection
            top = df.iloc[0]
            status = compliance_map.get(top['Object'], 'Unknown')
            if "Missing" in status:
                st.error(f"VIOLATION: {status}")
            elif "Compliant" in status:
                st.success(f"{status}")
            else:
                st.info(f"{status}")

            # Show results
            st.success(f"Top Detection: **{top['Object']}** ({top['Confidence']})")

            st.markdown("### All Detections")
            st.dataframe(df, use_container_width=True)

            # Summary
            st.markdown("### Summary")
            counts = df['Object'].value_counts()
            for obj, cnt in counts.items():
                icon = compliance_map.get(obj, "")
                st.write(f"{icon} **{obj}**: {cnt}")

            # Annotated image
            annotated = results.plot()
            st.image(annotated, caption="Detections", use_column_width=True)

        else:
            st.warning("No objects detected. Try a clearer image.")

else:
    st.info("Please upload an image to start.")

# Footer
st.markdown("---")
st.caption("Built with Ultralytics YOLOv5 + Streamlit | Works on Streamlit Cloud India")
