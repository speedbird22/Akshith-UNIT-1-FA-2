# app.py - Safety Compliance Detector (YOLOv5 + Streamlit Cloud Ready)
import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import pandas as pd

# === FIX NUMPY DEPRECATION WARNINGS ON STREAMLIT CLOUD ===
np.object = object
np.int = int
np.float = float
np.bool = bool

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Safety Compliance Detector",
    page_icon="Hard Hat",
    layout="centered"
)

# === TITLE & DESCRIPTION ===
st.title("Hard Hat Safety Compliance Detector")
st.markdown("""
Upload an image to instantly detect **workers**, **safety gear**, and **compliance status**  
(✅ Hardhat | ✅ Vest | ✅ Mask). Also detects vehicles, machinery & cones.
""")

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

# === LOAD MODEL (cached for speed) ===
@st.cache_resource(show_spinner="Loading YOLOv5 model...")
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.4  # confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.info("Make sure 'best.pt' is in the same folder as app.py")
        return None

model = load_model()

# === ASSOCIATION FUNCTION (gear belongs to person?) ===
def is_associated(person_box, gear_box, threshold=0.5):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = gear_box
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return (inter_area / gear_area) > threshold if gear_area > 0 else False

# === FILE UPLOADER ===
uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects..."):
        # Run inference
        results = model(img_cv, size=640)
        df = results.pandas().xyxy[0]  # DataFrame with detections

    # Draw results
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        label = row['name']
        conf = row.confidence
        mapped = compliance_map.get(label, label)
        color = (0, 255, 0) if "Compliant" in mapped or "Worker" in mapped else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_cv, f"{mapped} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)

    # === WORKER COMPLIANCE ANALYSIS ===
    persons = df[df['name'] == 'Person']
    st.subheader(f"Detected Workers: {len(persons)}")

    if len(persons) > 0:
        st.markdown("### Compliance Report")
        worker_reports = []

        for idx, person in persons.iterrows():
            pbox = (person.xmin, person.ymin, person.xmax, person.ymax)
            status = {"Hardhat": "Unknown", "Vest": "Unknown", "Mask": "Unknown"}

            # Hardhat
            if any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'Hardhat'].iterrows()):
                status["Hardhat"] = "Compliant"
            elif any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'NO-Hardhat'].iterrows()):
                status["Hardhat"] = "Missing Hardhat"

            # Vest
            if any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'Safety Vest'].iterrows()):
                status["Vest"] = "Compliant"
            elif any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'NO-Safety Vest'].iterrows()):
                status["Vest"] = "Missing Vest"

            # Mask
            if any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'Mask'].iterrows()):
                status["Mask"] = "Compliant"
            elif any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for _, r in df[df['name'] == 'NO-Mask'].iterrows()):
                status["Mask"] = "Missing Mask"

            worker_reports.append(status)

        # Display reports
        for i, report in enumerate(worker_reports, 1):
            with st.expander(f"Worker {i} Details"):
                st.write(f"Hardhat: **{report['Hardhat']}**")
                st.write(f"Safety Vest: **{report['Vest']}**")
                st.write(f"Mask: **{report['Mask']}**")

        # Summary
        total_compliant = sum(1 for r in worker_reports if all(v == "Compliant" for v in r.values() if v != "Unknown"))
        st.success(f"**{total_compliant}/{len(persons)} workers are fully compliant!**")

    # === OTHER OBJECTS ===
    others = df[df['name'].isin(['machinery', 'vehicle', 'Safety Cone'])]
    if not others.empty:
        st.subheader("Other Objects Detected")
        counts = others['name'].value_counts()
        for name, count in counts.items():
            icon = compliance_map.get(name, "")
            st.write(f"{icon} {name}: **{count}**")

else:
    if uploaded_file is None:
        st.info("Please upload an image to start detection.")
    else:
        st.error("Model not loaded. Check if 'best.pt' is in the root folder.")

# === FOOTER ===
st.markdown("---")
st.caption("YOLOv5 Custom Safety Gear Detector • Deployed on Streamlit Cloud • Made for India Construction Sites")
