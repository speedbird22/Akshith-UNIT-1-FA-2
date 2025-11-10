import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# Compliance mapping
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

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# IoU calculation (not used directly, but for reference if needed)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# Function to check if a small box (gear/violation) is mostly overlapping with person box
def is_associated(person_box, item_box, threshold=0.5):
    px1, py1, px2, py2 = person_box
    ix1, iy1, ix2, iy2 = item_box
    x1 = max(px1, ix1)
    y1 = max(py1, iy1)
    x2 = min(px2, ix2)
    y2 = min(py2, iy2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    item_area = (ix2 - ix1) * (iy2 - iy1) + 1e-6
    return (inter / item_area) > threshold

# Streamlit app
st.title('Safety Compliance Detector')
st.markdown('Upload an image to detect workers and check their compliance with safety gear (hardhat, safety vest, mask). The app also detects machinery, vehicles, and safety cones.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Perform detection
    results = model(img)
    
    # Get detections as pandas dataframe
    df = results.pandas().xyxy[0]
    
    # Draw bounding boxes with mapped labels
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']
        mapped_label = compliance_map.get(label, label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{mapped_label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display detected image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Detected Objects', use_column_width=True)
    
    # Extract persons
    persons = df[df['name'] == 'Person']
    num_workers = len(persons)
    st.subheader(f'Detected Workers: {num_workers}')
    
    if num_workers > 0:
        st.subheader('Compliance Report')
        compliance_list = []
        
        for idx, person in persons.iterrows():
            p_x1, p_y1, p_x2, p_y2 = person['xmin'], person['ymin'], person['xmax'], person['ymax']
            person_box = (p_x1, p_y1, p_x2, p_y2)
            
            person_compliance = {
                'Hardhat': '❓ Unknown',
                'Safety Vest': '❓ Unknown',
                'Mask': '❓ Unknown'
            }
            
            # Check Hardhat
            hardhats = df[df['name'] == 'Hardhat']
            no_hardhats = df[df['name'] == 'NO-Hardhat']
            has_hardhat = any(is_associated(person_box, (h['xmin'], h['ymin'], h['xmax'], h['ymax'])) for _, h in hardhats.iterrows())
            has_no_hardhat = any(is_associated(person_box, (nh['xmin'], nh['ymin'], nh['xmax'], nh['ymax'])) for _, nh in no_hardhats.iterrows())
            if has_hardhat:
                person_compliance['Hardhat'] = '✅ Compliant'
            elif has_no_hardhat:
                person_compliance['Hardhat'] = '❌ Missing Hardhat'
            
            # Check Safety Vest
            vests = df[df['name'] == 'Safety Vest']
            no_vests = df[df['name'] == 'NO-Safety Vest']
            has_vest = any(is_associated(person_box, (v['xmin'], v['ymin'], v['xmax'], v['ymax'])) for _, v in vests.iterrows())
            has_no_vest = any(is_associated(person_box, (nv['xmin'], nv['ymin'], nv['xmax'], nv['ymax'])) for _, nv in no_vests.iterrows())
            if has_vest:
                person_compliance['Safety Vest'] = '✅ Compliant'
            elif has_no_vest:
                person_compliance['Safety Vest'] = '❌ Missing Vest'
            
            # Check Mask
            masks = df[df['name'] == 'Mask']
            no_masks = df[df['name'] == 'NO-Mask']
            has_mask = any(is_associated(person_box, (m['xmin'], m['ymin'], m['xmax'], m['ymax'])) for _, m in masks.iterrows())
            has_no_mask = any(is_associated(person_box, (nm['xmin'], nm['ymin'], nm['xmax'], nm['ymax'])) for _, nm in no_masks.iterrows())
            if has_mask:
                person_compliance['Mask'] = '✅ Compliant'
            elif has_no_mask:
                person_compliance['Mask'] = '❌ Missing Mask'
            
            compliance_list.append(person_compliance)
        
        # Display compliance in a nice format
        for i, comp in enumerate(compliance_list, 1):
            with st.expander(f"Worker {i} Compliance Details"):
                st.write(f"- Hardhat: {comp['Hardhat']}")
                st.write(f"- Safety Vest: {comp['Safety Vest']}")
                st.write(f"- Mask: {comp['Mask']}")
    
    # Other detections
    other_classes = ['machinery', 'vehicle', 'Safety Cone']
    others = df[df['name'].isin(other_classes)]
    if not others.empty:
        st.subheader('Other Detected Objects')
        other_counts = others['name'].value_counts()
        for obj, count in other_counts.items():
            mapped = compliance_map.get(obj, obj)
            st.write(f"{mapped}: {count}")
