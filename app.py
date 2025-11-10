# app.py
import streamlit as st
from PIL import Image
import pandas as pd
import os

# MAGIC LINE #1 — DISABLE OPENCV BEFORE ANYTHING
os.environ["ULTRALYTICS_DISABLE_OPENCV"] = "1"
os.environ["YOLO_DISABLE_OPENCV"] = "1"

# MAGIC LINE #2 — FORCE NO GUI
os.environ["MPLBACKEND"] = "Agg"

# NOW IMPORT YOLO SAFELY
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="PPE Detector", page_icon="hardhat", layout="centered")

st.title("hardhat Construction PPE Compliance Checker")
st.caption("Upload image → Instant safety report")

# Load model
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

# Compliance status
status_map = {
    'Hardhat': 'Compliant - Hardhat',
    'Safety Vest': 'Compliant - Vest',
    'Mask': 'Compliant - Mask',
    'NO-Hardhat': 'VIOLATION: Missing Hardhat',
    'NO-Safety Vest': 'VIOLATION: Missing Vest',
    'NO-Mask': 'VIOLATION: Missing Mask',
    'Person': 'Worker Detected',
    'machinery': 'Machinery',
    'vehicle': 'Vehicle',
    'Safety Cone': 'Cone'
}

# Upload
uploaded = st.file_uploader("Upload construction photo", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, "Your Image")

    with st.spinner("Scanning for PPE..."):
        results = model(img, conf=0.25, imgsz=640, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            names = results.names

            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                name = names[cls_id]
                detections.append({"Item": name, "Confidence": f"{conf:.1%}"})

            df = pd.DataFrame(detections).sort_values("Confidence", ascending=False)

            # Show top violation
            top_item = df.iloc[0]["Item"]
            top_msg = status_map.get(top_item, top_item)
            if "VIOLATION" in top_msg:
                st.error(top_msg)
            else:
                st.success(top_msg)

            st.write("### All Objects Detected")
            st.dataframe(df)

            # Show image with boxes
            annotated = results.plot()
            st.image(annotated, "Detections on Image", use_column_width=True)

            # Summary
            st.write("### Summary")
            for item, count in df["Item"].value_counts().items():
                st.write(f"• {status_map.get(item, item)} → **{count}x**")

        else:
            st.warning("No objects detected. Try a clearer image with workers.")

else:
    st.info("Upload an image to check PPE compliance")

st.markdown("---")
st.caption("Works 100% on Streamlit Cloud India | No OpenCV | Nov 10, 2025")
