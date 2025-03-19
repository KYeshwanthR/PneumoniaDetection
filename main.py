import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
import cv2

device = torch.device("cpu")
model = torchvision.models.efficientnet_v2_l(weights=None)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

try:
    model_weights_path = 'WeightsFile.pth'
    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
except Exception as e:
    st.error(f"Error loading model weights: {str(e)}")
    st.stop()

model = model.to(device)
model.eval()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, np.array(img) / 255.0

def predict_pneumonia(image_path, model, device):
    input_tensor, rgb_img = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities, rgb_img, input_tensor

def visualize_gradcam(model, input_tensor, rgb_img, predicted_class):
    target_layers = [model.features[-1][-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

    visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)
    return visualization

st.title("Pneumonia Detection")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            image_path = tmp_file.name

        st.image(image_path, caption="Uploaded Image",use_column_width=True)

        with st.spinner("Analyzing the image..."):
            predicted_class, probabilities, rgb_img, input_tensor = predict_pneumonia(image_path, model, device)
            prediction = "Pneumonia" if predicted_class == 1 else "Normal"

            st.write(f"### Prediction: {prediction}")
            st.write(f"#### Probabilities:")
            st.write(f"- Normal: {probabilities[0]:.4f}")
            st.write(f"- Pneumonia: {probabilities[1]:.4f}")

            visualization = visualize_gradcam(model, input_tensor, rgb_img, predicted_class)

            st.image(visualization, caption="Grad-CAM Visualization",use_column_width=True)

        os.unlink(image_path)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")