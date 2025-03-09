import os
import io
import base64
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from flask import Flask, request, render_template, redirect, url_for

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

results_history = []
next_id = 0

model = torchvision.models.efficientnet_v2_l(weights=None)
num_ftrs = model.classifier[1].in_features
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

model_weights_path = 'WeightsFile.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=True))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def preprocess_image(file_obj):
    img = Image.open(file_obj).convert('RGB')
    original_img = img.copy()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    rgb_img = np.array(original_img) / 255.0
    return input_tensor, rgb_img, original_img

def predict_pneumonia(file_obj, model, device):
    input_tensor, rgb_img, original_img = preprocess_image(file_obj)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities, rgb_img, input_tensor, original_img

def visualize_gradcam(model, input_tensor, rgb_img, predicted_class):
    target_layers = [model.features[-1][-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
    visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=False)
    return visualization

def convert_array_to_base64(image_array):
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    retval, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/', methods=['GET', 'POST'])
def index():
    global results_history, next_id

    sample_folder = os.path.join(app.static_folder, "sample")
    sample_images = []
    if os.path.exists(sample_folder):
        for img in os.listdir(sample_folder):
            if img.startswith("._"):
                continue
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                sample_images.append(img)

    if request.method == 'POST':
        uploaded_files = request.files.getlist("files")
        new_results = []
        for file in uploaded_files:
            if file:
                predicted_class, probabilities, rgb_img, input_tensor, original_img = predict_pneumonia(file, model, device)
                visualization = visualize_gradcam(model, input_tensor, rgb_img, predicted_class)
                original_img_b64 = pil_to_base64(original_img)
                visualization_b64 = convert_array_to_base64(visualization)
                prediction = "Pneumonia" if predicted_class == 1 else "Normal"
                new_results.append({
                    "id": next_id,
                    "original": original_img_b64,
                    "visualization": visualization_b64,
                    "prediction": prediction,
                    "probabilities": {
                        "Normal": f"{probabilities[0]:.4f}",
                        "Pneumonia": f"{probabilities[1]:.4f}"
                    }
                })
                next_id += 1

        results_history = new_results + results_history
        return redirect(url_for('index'))

    return render_template("index.html", results=results_history, samples=sample_images)

@app.route('/remove/<int:image_id>', methods=['GET'])
def remove_image(image_id):
    global results_history
    results_history = [r for r in results_history if r['id'] != image_id]
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
