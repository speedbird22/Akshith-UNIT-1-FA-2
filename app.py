# app.py - NO CV2, 100% WORKING ON STREAMLIT CLOUD (Nov 10, 2025)
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import pandas as pd

# Fix NumPy deprecation
np.object = object
np.int = int
np.float = float
np.bool = bool

st.set_page_config(page_title="Safety Detector", page_icon="Hard Hat", layout="centered")

st.title("Construction Site Safety Compliance Detector")
st.markdown("**No OpenCV • Pure PIL • Zero Errors • Lightning Fast**")

# Compliance map with emojis
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

@st.cache_resource(show_spinner="Loading YOLOv5 model...")
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()
model.conf = 0.40
model.iou = 0.45

def is_associated(person_box, gear_box, threshold=0.6):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = gear_box
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return inter / gear_area > threshold if gear_area > 0 else False

def draw_boxes_pil(image, df):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
        label = compliance_map.get(row.name, row.name)
        conf = row.confidence
        text = f"{label} {conf:.2f}"
        
        # Color: Green for compliant/worker, Red for missing
        color = "lime" if any(x in label for x in ["Compliant", "Worker"]) else "red"
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1-30), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x1+5, y1-30), text, fill="black", font=font)
    
    return image

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Load image with PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy for YOLO
    img_np = np.array(image)

    with st.spinner("Detecting..."):
        results = model(img_np, size=640)
        df = results.pandas().xyxy[0]

    # Draw using PIL only
    result_image = draw_boxes_pil(image.copy(), df)
    st.image(result_image, caption="Detection Results", use_column_width=True)

    # Workers
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
                has = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax))
                         for r in df[df['name'] == gear].itertuples())
                missing = any(is_associated(pbox, (r.xmin, r.ymin, r.xmax, r.ymax))
                             for r in df[df['name'] == no_gear].itertuples())
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

    # Other objects
    others = df[df['name'].isin(['machinery', 'vehicle', 'Safety Cone'])]
    if not others.empty:
        st.subheader("Other Objects")
        for obj in others['name'].unique():
            count = len(others[others['name'] == obj])
            icon = compliance_map.get(obj, "")
            st.write(f"{icon} {obj}: **{count}**")

else:
    st.info("Upload an image to start")

st.caption("Zero CV2 • Pure PIL • Made for India • Nov 10, 2025")
