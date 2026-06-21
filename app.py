import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
 
IMG_SIZE = 150
MODEL_PATH = "brain_tumor_model_final.keras"
CLASS_INDICES_PATH = "class_indices.json"
 
 
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
 
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
 
    # class_indices looks like {"glioma": 0, "meningioma": 1, ...}
    # we need the reverse mapping: index -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class
 
 
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Match the exact preprocessing used during training."""
    img = pil_image.convert("RGB")               # force 3 channels, RGB order
    img = img.resize((IMG_SIZE, IMG_SIZE))        # same size as training
    arr = np.array(img, dtype=np.float32) / 255.0  # same scaling as training
    arr = np.expand_dims(arr, axis=0)             # add batch dimension
    return arr
 
 
def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")
    st.title("🧠 Brain Tumor MRI Classifier")
 
    model, idx_to_class = load_assets()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
 
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image", type=["jpg", "jpeg", "png"]
    )
 
    if uploaded_file is None:
        st.info("Ek MRI image upload karein prediction dekhne ke liye.")
        return
 
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
 
    input_tensor = preprocess_image(pil_image)
    probs = model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])
 
    st.success(f"🎯 **Prediction:** {pred_label}  (confidence: {confidence:.2%})")
 
    st.subheader("Raw Model Output")
    st.table({name: [f"{p:.6f}"] for name, p in zip(class_names, probs)})
 
 
if __name__ == "__main__":
    main()