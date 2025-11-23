import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

# --------------------------------------------------
# 1. Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 2. Load MobileNetV2 (Classification)
# --------------------------------------------------
model_cls = models.mobilenet_v2(weights=None)
model_cls.classifier[1] = torch.nn.Linear(1280, 2)
model_cls.load_state_dict(torch.load("best_mobilenet_model.pth", map_location=device))
model_cls.to(device)
model_cls.eval()

# Classification transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class_names = ["bird", "drone"]

# --------------------------------------------------
# 3. Load YOLOv8 Model (Detection)
# --------------------------------------------------
yolo = YOLO("best_yolo_model.pt")

# --------------------------------------------------
# 4. Prediction Functions
# --------------------------------------------------
def classify_mobilenet(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_cls(tensor)
        prob = F.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
    return class_names[pred.item()], conf.item() * 100

def detect_yolo(image):
    results = yolo.predict(image, conf=0.25, verbose=False)
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    detections_info = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = f"{yolo.names[cls_id]} {conf:.2f}"
            colour = (0, 255, 0) if cls_id == 0 else (0, 0, 255)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img_cv, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            detections_info.append((yolo.names[cls_id], conf))

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_rgb, detections_info

# --------------------------------------------------
# 5. Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Bird vs Drone AI Detector", layout="centered")

st.title("🐦🚁 Bird vs Drone – Detection & Classification")
st.write("Upload an image. YOLOv8 will detect objects and MobileNetV2 will classify the image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Center image
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Running YOLO Detection..."):
        yolo_img, detections = detect_yolo(image)

    with st.spinner("Running MobileNet Classification..."):
        label, conf = classify_mobilenet(image)

    st.markdown("---")

    # YOLO results
    st.subheader("📦 YOLOv8 Detection Results (With Bounding Boxes)")
    st.image(yolo_img, width=350)

    if detections:
        st.write("### Detected Objects:")
        for cls, c in detections:
            st.write(f"**{cls.capitalize()}** — {c*100:.2f}% confidence")
    else:
        st.warning("No objects detected.")

    # MobileNet results
    st.markdown("---")
    st.subheader("🧠 MobileNetV2 Classification Result")
    st.success(f"**Class:** {label.capitalize()}")
    st.info(f"**Confidence:** {conf:.2f}%")
