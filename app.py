# app.py - FINAL WORKING VERSION FOR STREAMLIT CLOUD (November 10, 2025)
import streamlit as st
from PIL import Image
import torch
import numpy as np
import pandas as pd

# === CRITICAL FIX FOR STREAMLIT CLOUD ===
import sys
import os
sys.path.append('/usr/local/lib/python3.11/site-packages')
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**64)

# Now safe to import cv2
import cv2

# Fix NumPy deprecation (required on Streamlit Cloud)
np.object = object
np.int = int
np.float = float
np.bool = bool

# === PAGE CONFIG ===
st.set_page_config(page_title="Safety Gear Detector", page_icon="Hard Hat", layout="centered")

# === HEADER ===
st.title("Hard Hat Safety Gear & Compliance Detector")
st.markdown("**Upload image → Detect workers → Instant compliance report**")

# === COMPLIANCE MAP ===
compliance_map = {
    'Hardhat': 'Compliant',
    'Safety Vest': 'Compliant',
    'Mask': 'Compliant',
    'NO-Hardhat': 'Missing Hardhat',
    'NO-Safety Vest': 'Missing Vest',
    'NO-Mask': 'Missing Mask',
    'Person': 'Worker',
    'machinery': 'Machinery',
    'vehicle': 'Vehicle',
    'Safety Cone': 'Cone'
}

# === LOAD MODEL ===
@st.cache_resource(show_spinner="Loading YOLOv5 model (this takes ~20 seconds first time)...")
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.40
        model.iou = 0.45
        return model
    except Exception as e:
        st.error("Model failed to load. Make sure 'best.pt' is in the repo root.")
        st.error(f"Error: {e}")
        return None

model = load_model()

# === ASSOCIATION LOGIC ===
def is_associated(person_box, gear_box, threshold=0.6):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = gear_box
    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return inter_area / gear_area > threshold if gear_area > 0 else False

# === UPLOADER ===
uploaded_file = st.file_uploader("Upload Construction Site Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        results = model(img_cv, size=640)
        df = results.pandas().xyxy[0]

    # Draw boxes
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
        label = compliance_map.get(row.name, row.name)
        conf = row.confidence
        color = (0, 255, 0) if "Compliant" in label or "Worker" in label else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)

    # === COMPLIANCE REPORT ===
    persons = df[df['name'] == 'Person']
    st.subheader(f"Found {len(persons)} Worker(s)")

    if len(persons) > 0:
        st.markdown("### Compliance Status")
        fully_compliant = 0
        for i, person in enumerate(persons.itertuples(), 1):
            pbox = (person.xmin, person.ymin, person.xmax, person.ymax)
            status = {"Hardhat": "Unknown", "Vest": "Unknown", "Mask": "Unknown"}

            # Check gear
            for gear_class, yes_no in [
                ('Hardhat', 'NO-Hardhat'),
                ('Safety Vest', 'NO-Safety Vest'),
                ('Mask', 'NO-Mask')
            ]:
                has_gear = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) 
                             for r in df[df['name'] == gear_class].itertuples())
                no_gear = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) 
                            for r in df[df['name'] == yes_no].itertuples())
                if has_gear:
                    status[gear_class.split()[-1]] = "Compliant"
                elif no_gear:
                    status[gear_class.split()[-1]] = "Missing " + gear_class.split()[-1]

            with st.expander(f"Worker {i}"):
                st.write(f"Hardhat: **{status['Hardhat']}**")
                st.write(f"Safety Vest: **{status['Vest']}**")
                st.write(f"Mask: **{status['Mask']}**")

            if all(v == "Compliant" for v in status.values()):
                fully_compliant += 1

        st.success(f"**{fully_compliant}/{len(persons)} workers fully compliant!**")

    # Other objects
    others = df[df['name'].isin(['machinery', 'vehicle', 'Safety Cone'])]
    if not others.empty:
        st.subheader("Other Objects")
        for obj in others['name'].unique():
            count = len(others[others['name'] == obj])
            icon = compliance_map.get(obj, "")
            st.write(f"{icon} {obj}: **{count}**")

else:
    st.info("Upload an image to start detection")

st.caption("Made for Indian Construction Sites | YOLOv5 + Streamlit Cloud | Nov 2025")
