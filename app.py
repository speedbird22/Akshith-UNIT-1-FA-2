import streamlit as st
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load model
model = YOLO("best.pt")

# Class categories (must match your training)
CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
           'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Compliance logic
def assess_compliance(results):
    df = results.pandas().xyxy[0]
    summary = {"total": 0, "compliant": 0, "partial": 0, "non_compliant": 0}
    alerts = []

    persons = df[df['name'] == 'Person']
    summary["total"] = len(persons)

    for _, person in persons.iterrows():
        x1, y1, x2, y2 = person[['xmin', 'ymin', 'xmax', 'ymax']]
        ppe_items = df[
            (df['xmin'] > x1) & (df['xmax'] < x2) &
            (df['ymin'] > y1) & (df['ymax'] < y2)
        ]['name'].tolist()

        if all(p in ppe_items for p in ['Hardhat', 'Mask', 'Safety Vest']):
            summary["compliant"] += 1
        elif any(p in ppe_items for p in ['Hardhat', 'Mask', 'Safety Vest']):
            summary["partial"] += 1
            alerts.append("⚠️ Partial compliance detected.")
        else:
            summary["non_compliant"] += 1
            alerts.append("❌ Worker without PPE detected.")

    return summary, alerts, df

# Streamlit UI
st.set_page_config(page_title="PPE Compliance Dashboard", layout="wide")
st.title("👷 PPE Compliance Dashboard")
st.markdown("Upload an image to detect workers and assess safety compliance.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv5 inference
    results = model(image)
    summary, alerts, detections = assess_compliance(results)

    # Display summary
    st.subheader("Detection Summary")
    st.metric("Total Workers", summary["total"])
    st.metric("✅ Compliant", summary["compliant"])
    st.metric("⚠️ Partially Compliant", summary["partial"])
    st.metric("❌ Non-Compliant", summary["non_compliant"])

    # Alerts
    if alerts:
        st.subheader("🚨 Violation Alerts")
        for alert in alerts:
            st.error(alert)

    # Chart
    st.subheader("Compliance Breakdown")
    fig, ax = plt.subplots()
    ax.bar(["Compliant", "Partial", "Non-Compliant"], 
           [summary["compliant"], summary["partial"], summary["non_compliant"]],
           color=["green", "orange", "red"])
    st.pyplot(fig)

    # Table
    st.subheader("Detection Details")
    st.dataframe(detections)
