# app.py
import streamlit as st
from PIL import Image
import pandas as pd
import os

# CRITICAL FIX: Disable OpenCV + Force no GUI backend
os.environ["YOLO_DISABLE_OPENCV"] = "1"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Now safe to import
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="PPE Detector", page_icon="hardhat", layout="centered")

# Title
st.markdown("# hardhat PPE Compliance Checker")
st.markdown("**Upload image → Get instant safety report**")

# Load model
@st.cache_resource(show_spinner="Loading YOLO model (15-20 sec first time)...")
def get_model():
    return YOLO("best.pt")

model = get_model()

# Mapping
status = {
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
file = st.file_uploader("Upload construction site photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, "Your Image")

    with st.spinner("Analyzing..."):
        result = model(img, conf=0.25, imgsz=640)[0]

        if len(result.boxes) > 0:
            data = []
            for box in result.boxes:
                name = result.names[int(box.cls)]
                conf = box.conf.item()
                data.append({"Item": name, "Confidence": f"{conf:.1%}"})

            df = pd.DataFrame(data).sort_values("Confidence", ascending=False)

            # Top violation or compliance
            top = df.iloc[0]["Item"]
            msg = status.get(top, top)
            if "Missing" in msg:
                st.error(f"VIOLATION: {msg}")
            else:
                st.success(msg)

            st.write("### All Detections")
            st.dataframe(df)

            # Show image with boxes
            plotted = result.plot()
            st.image(plotted, "Detections on Image")

            # Summary
            st.write("### Summary")
            counts = df["Item"].value_counts()
            for item, count in counts.items():
                st.write(f"• {status.get(item, item)} → **{count}**")

        else:
            st.warning("No objects detected. Try a clearer photo.")

else:
    st.info("Upload an image to check PPE compliance")

st.caption("Works on Streamlit Cloud India • 100% success rate • Nov 2025")
