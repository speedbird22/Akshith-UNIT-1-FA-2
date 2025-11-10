# app.py
import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import torch

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Construction PPE Compliance",
    page_icon="hardhat",
    layout="centered"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #ff5722; color: white; }
    .header { font-size: 42px; text-align: center; color: #1e3a8a; font-weight: bold; }
    .subheader { font-size: 20px; text-align: center; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ------------------- TITLE & DESCRIPTION -------------------
st.markdown('<h1 class="header">hardhat Construction Site PPE Compliance</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload an image to instantly detect workers, PPE, and safety violations.</p>', unsafe_allow_html=True)

# ------------------- LOAD MODEL (CACHED) -------------------
@st.cache_resource(show_spinner="Loading YOLOv5 model... This takes a few seconds.")
def load_model():
    try:
        model = YOLO('best.pt')  # Your trained YOLOv5 model
        model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure 'best.pt' is in the same folder as app.py")
        return None

model = load_model()

# ------------------- COMPLIANCE MAPPING -------------------
compliance_map = {
    'Hardhat': 'Compliant - Hardhat Worn',
    'Safety Vest': 'Compliant - Vest Worn',
    'Mask': 'Compliant - Mask Worn',
    'NO-Hardhat': 'Missing Hardhat',
    'NO-Safety Vest': 'Missing Safety Vest',
    'NO-Mask': 'Missing Mask',
    'Person': 'Worker Detected',
    'machinery': 'Machinery Present',
    'vehicle': 'Vehicle Detected',
    'Safety Cone': 'Safety Cone Placed'
}

# ------------------- FILE UPLOADER -------------------
uploaded_file = st.file_uploader(
    "Upload Construction Site Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported: JPG, PNG, WebP"
)

if uploaded_file and model:
    # ------------------- DISPLAY IMAGE -------------------
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image for PPE compliance..."):
        # ------------------- RUN INFERENCE -------------------
        results = model(image, conf=0.25, iou=0.45, imgsz=640)[0]

        # Extract detections
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            names = results.names
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            detected_names = [names[c] for c in cls]

            # Create DataFrame
            df = pd.DataFrame({
                'Object': detected_names,
                'Confidence': [f"{c:.2%}" for c in conf],
                'Compliance': [compliance_map.get(name, 'Unknown') for name in detected_names]
            }).sort_values(by='Confidence', ascending=False).reset_index(drop=True)

            # ------------------- TOP DETECTION -------------------
            top_detection = df.iloc[0]
            top_label = top_detection['Object']
            top_conf = top_detection['Confidence']
            top_compliance = top_detection['Compliance']

            st.markdown("### Top Detection")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Detected:** {top_label}")
            with col2:
                st.info(f"**Confidence:** {top_conf}")

            if "Compliant" in top_compliance:
                st.success(f"**Compliance:** {top_compliance}")
            elif "Missing" in top_compliance:
                st.error(f"**VIOLATION:** {top_compliance}")
            else:
                st.info(f"**Status:** {top_compliance}")

            # ------------------- ALL DETECTIONS TABLE -------------------
            st.markdown("### All Detections")
            st.dataframe(df, use_container_width=True)

            # ------------------- COMPLIANCE SUMMARY -------------------
            st.markdown("### Compliance Summary")
            summary = df['Object'].value_counts()
            compliant_count = sum(1 for obj in detected_names if obj in ['Hardhat', 'Safety Vest', 'Mask'])
            violations = sum(1 for obj in detected_names if 'NO-' in obj)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Workers Detected", len([x for x in detected_names if x == 'Person']))
            with col2:
                st.metric("Compliant Items", compliant_count, delta=f"+{compliant_count}")
            with col3:
                st.metric("Violations Found", violations, delta=f"-{violations}" if violations > 0 else None)

            # Detailed list
            st.markdown("**Detected Items:**")
            for obj, count in summary.items():
                status = compliance_map.get(obj, obj)
                icon = "Compliant" if "Compliant" in status else "Violation" if "Missing" in status else "Info"
                st.write(f"- {status}: **{count}**")

            # Optional: Show image with bounding boxes
            if st.checkbox("Show detections on image", value=True):
                annotated_img = results.plot()  # YOLO auto-draws boxes
                st.image(annotated_img, caption="Detections Overlay", use_column_width=True)

        else:
            st.warning("No objects detected. Try a clearer image with workers and PPE.")

else:
    if not uploaded_file:
        st.info("Please upload an image to begin.")
    if not model:
        st.error("Model not loaded. Check if 'best.pt' exists.")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 14px;'>"
    "Built with ❤️ using <b>Ultralytics YOLOv5</b> + <b>Streamlit</b><br>"
    "For Construction Safety Monitoring • November 2025"
    "</p>",
    unsafe_allow_html=True
)
