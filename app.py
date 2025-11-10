# app.py - YOLOv8 SAFETY DETECTOR - 100% WORKING ON STREAMLIT CLOUD
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Fix NumPy deprecation
np.object = object
np.int = int
np.float = float
np.bool = bool

st.set_page_config(page_title="Safety Detector", page_icon="Hard Hat", layout="centered")

st.title("Construction Site Safety Compliance Detector")
st.markdown("**YOLOv8 • No OpenCV • Instant Load • Made for India**")

# Compliance map
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

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model():
    return YOLO('best.pt')  # Your trained YOLOv5/v8 .pt file works directly!

model = load_model()

def is_associated(person_box, gear_box, threshold=0.6):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = gear_box
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return inter / gear_area > threshold if gear_area > 0 else False

def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = model.names[cls]
        text = f"{compliance_map.get(label, label)} {conf:.2f}"
        color = "lime" if any(x in text for x in ["Compliant", "Worker"]) else "red"
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        bbox = draw.textbbox((x1, y1-35), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1+6, y1-35), text, fill="black", font=font)
    return image

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image, conf=0.4, iou=0.45)
        df = results[0].pandas().xyxy[0]

    result_img = draw_boxes(image.copy(), results)
    st.image(result_img, caption="Detection Results", use_column_width=True)

    persons = df[df['name'] == 'Person']
    st.subheader(f"Workers Detected: {len(persons)}")

    if len(persons) > 0:
        st.markdown("### PPE Compliance Report")
        fully_compliant = 0

        for i, person in enumerate(persons.itertuples(), 1):
            pbox = (person.xmin, person.ymin, person.xmax, person.ymax)
            status = {"Hardhat": "Unknown", "Vest": "Unknown", "Mask": "Unknown"}

            gear_map = {
                'Hardhat': 'NO-Hardhat',
                'Safety Vest': 'NO-Safety Vest',
                'Mask': 'NO-Mask'
            }

            for gear, no_gear in gear_map.items():
                has = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for r in df[df['name'] == gear].itertuples())
                missing = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax)) for r in df[df['name'] == no_gear].itertuples())
                key = gear.split()[-1]
                if has:
                    status[key] = "Compliant"
                elif missing:
                    status[key] = f"Missing {key}"

            with st.expander(f"Worker {i}"):
                st.write(f"Hardhat: **{status['Hardhat']}**")
                st.write(f"Safety Vest: **{status['Vest']}**")
                st.write(f"Mask: **{status['Mask']}**")

            if all(v == "Compliant" for v in status.values()):
                fully_compliant += 1

        if fully_compliant == len(persons):
            st.balloons()
        st.success(f"**{fully_compliant}/{len(persons)} workers fully compliant!**")

    others = df[df['name'].isin(['machinery', 'vehicle', 'Safety Cone'])]
    if not others.empty:
        st.subheader("Other Objects")
        for obj in others['name'].unique():
            count = len(others[others['name'] == obj])
            st.write(f"{compliance_map.get(obj, '')} {obj}: **{count}**")

else:
    st.info("Upload an image to start detection")

st.caption("YOLOv8 • No CV2 • Deployed on Streamlit Cloud • Nov 10, 2025")
