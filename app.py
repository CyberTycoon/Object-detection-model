

import torch
# Create YOLO-style image batches
batch = torch.zeros((16, 3, 640, 640), dtype=torch.float16)  # YOLO-native size
# Simulate RAM constraints
with torch.autocast('cpu'):  # Even on CPU!
    processed = batch * 255
print(f"Memory saved: {batch.element_size() * batch.nelement() / 1e6}MB → {processed.element_size() * processed.nelement() / 1e6}MB")


# In[14]:


bad_tensor = torch.zeros(256,256, dtype=torch.float64)  # 8 bytes per number
good_tensor = torch.zeros(256,256, dtype=torch.float16)  # 2 bytes per number

print(f"Memory saved: {bad_tensor.element_size() * bad_tensor.nelement() / 1e6}MB → {good_tensor.element_size() * good_tensor.nelement() / 1e6}MB")


# In[15]:


number_1 = torch.tensor([36.4, 36.7, 36.5, 36.6, 36.5], dtype=torch.float16)
number_2 = torch.tensor((3, 128, 128), dtype= torch.float16)
number_3 = torch.tensor((10, 1, 256, 256), dtype = torch.float16)


# In[16]:


def print_memory(tensor, name):
    bytes = tensor.element_size() * tensor.nelement()
    print(f"{name}: {bytes / 1024}KB")



print_memory(number_2, 'test')


# In[17]:


# 1. Temperature vector
temps = torch.tensor([36.4, 36.7, 36.5, 36.6, 36.5], dtype=torch.float16)

# 2. RGB image (3 channels, 128x128)
rgb_image = torch.zeros((3, 128, 128), dtype=torch.float16)

# 3. MRI slices (10 slices, 1 channel, 256x256)
mri_volume = torch.rand((10, 1, 256, 256), dtype=torch.float16)

# Memory calculations
print_memory(temps, "Temperatures")        # ~0.01KB
print_memory(rgb_image, "RGB Image")       # 3*128*128*2 = 98KB
print_memory(mri_volume, "MRI Volume")     # 10*1*256*256*2 = 1.25MB


# In[18]:

import streamlit as st 
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ======= MODEL SETUP =======
# Load the standard YOLOv8 model (downloads it if not present)
model = YOLO('yolov8n.pt')  # Standard YOLOv8 model with 80 COCO classes

# ======= STREAMLIT UI =======
st.title(" Object Detection")
st.write("Upload an image to detect objects from the 80 COCO classes")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Show processing indicator
    with st.spinner("Detecting objects..."):
        # Add small delay to show the spinner
        time.sleep(0.5)
        
        # Run inference with YOLOv8
        results = model.predict(img_array, conf=0.25)  # Set confidence threshold
    
    # Display results
    st.write("### Detection Results")
    
    # Get the results from the first image
    result = results[0]
    
    # Display the annotated image with bounding boxes
    annotated_img = result.plot()
    st.image(annotated_img, caption="Detection Results", use_column_width=True)
    
    # Show detection summary
    if result.boxes is not None and len(result.boxes) > 0:
        st.write(f"Detected {len(result.boxes)} objects:")
        
        # Create a dictionary to count objects by class
        class_counts = {}
        
        # Process each detection
        for box in result.boxes:
            # Get class name and confidence
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            confidence = box.conf[0].item()
            
            # Add to count
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
                
            # Get bounding box coordinates (optional)
            # bbox = box.xyxy[0].tolist()  # in (x1, y1, x2, y2) format
        
        # Display summary table
        st.write("#### Objects Detected")
        for class_name, count in class_counts.items():
            st.write(f"- {class_name}: {count}")
    else:
        st.write("No objects detected with confidence above threshold.")
        
    # Show model information
    with st.expander("Model Information"):
        st.write("""
        **Model:** YOLOv8n (nano variant)
        **Classes:** 80 COCO dataset classes
        **Framework:** Ultralytics YOLOv8
        
        This model can detect common objects like people, animals, vehicles, and everyday items.
        You can adjust the confidence threshold for more or fewer detections.
        """)