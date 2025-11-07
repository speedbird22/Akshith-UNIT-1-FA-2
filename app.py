import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load model manually
model = torch.load("best.pt", map_location=torch.device("cpu"))
model.eval()

# Streamlit UI
st.set_page_config(page_title="PPE Compliance Dashboard", layout="wide")
st.title("👷 PPE Compliance Dashboard")
st.markdown("Upload an image to detect workers and assess safety compliance.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to tensor
    img = np.array(image)
    img = img[:, :, ::-1]  # RGB to BGR
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        results = model(img)

    # Parse results (you may need to adjust this based on your model's output format)
    # For now, just show raw tensor
    st.subheader("Raw Detection Output")
    st.write(results)

    st.warning("⚠️ This version skips compliance logic and bounding boxes. Let me know if you want a custom parser.")
