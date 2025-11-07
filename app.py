import streamlit as st
import torch
from PIL import Image
import pandas as pd

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Map detected classes to compliance categories
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

# Streamlit UI
st.set_page_config(page_title="Construction PPE Dashboard", layout="wide")
st.title("👷 Construction Site PPE Compliance")
st.markdown("Upload an image to detect workers and assess safety compliance.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    detections = results.pandas().xyxy[0]

    # Extract top detection
    if not detections.empty:
        top = detections.iloc[0]
        label = top['name']
        conf = round(top['confidence'] * 100, 2)
        category = compliance_map.get(label, 'Unknown')

        # Display result
        st.subheader("Top Detection")
        st.success(f"Detected: {label}")
        st.info(f"Confidence: {conf}%")
        if category.startswith("✅"):
            st.success(f"Compliance: {category}")
        elif category.startswith("❌"):
            st.warning(f"Violation: {category}")
        else:
            st.info(f"Category: {category}")

    # Show full detection table
    st.subheader("All Detections")
    st.dataframe(detections)

    # Compliance summary
    st.subheader("Compliance Summary")
    summary = detections['name'].value_counts().to_dict()
    for cls, count in summary.items():
        label = compliance_map.get(cls, cls)
        st.write(f"- {label}: {count}")
