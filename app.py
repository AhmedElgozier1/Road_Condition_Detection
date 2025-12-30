import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

# =========================
# Device
# =========================
device = torch.device("cpu")

# =========================
# Load Trained Model
# =========================
def load_model():
    checkpoint = torch.load("road_condition_model.pth", map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print("Model loaded successfully")
    return model, class_names

model, class_names = load_model()

# =========================
# Class Display Names
# =========================
display_names = {
    "pothole": "Pothole",
    "road_damage": "Road Damage",
    "illegal_parking": "Illegal Parking",
    "other_issue": "Other Road Issue"
}

# =========================
# Image Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# =========================
# Prediction Function
# =========================
def predict_road_condition(image):
    if image is None:
        return (
            "<div style='text-align:center;color:red;'><h3>Please upload an image first</h3></div>",
            None
        )

    image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    idx = np.argmax(probs)
    raw_class = class_names[idx]
    predicted_class = display_names.get(raw_class, raw_class)
    confidence = probs[idx]

    detailed_results = {
        display_names.get(class_names[i], class_names[i]): float(probs[i])
        for i in range(len(class_names))
    }

    result_html = f"""
    <div style="padding:25px;border-radius:15px;
                border-right:6px solid #4f46e5;
                background:#f9f9f9;">
        <h2>Prediction Result</h2>
        <h1 style="color:#111;">{predicted_class}</h1>
        <h3 style="color:#16a34a;">Confidence: {confidence*100:.2f}%</h3>
        <p style="color:#555;">
            The prediction is generated using a deep learning model trained on real-world road images.
        </p>
    </div>
    """

    return result_html, detailed_results

# =========================
# GUI
# =========================
with gr.Blocks(title="Road Condition Detection System") as demo:

    gr.Markdown("""
    # Road Condition Detection System
    Deep Learning based image classification using PyTorch
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Road Image",
                type="numpy"
            )
            analyze_btn = gr.Button("Analyze Image")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            output_html = gr.HTML()
            confidence_plot = gr.Label(
                label="Class Probabilities",
                num_top_classes=4
            )

    analyze_btn.click(
        fn=predict_road_condition,
        inputs=image_input,
        outputs=[output_html, confidence_plot]
    )

    clear_btn.click(
        fn=lambda: (None, None),
        inputs=None,
        outputs=[output_html, confidence_plot]
    )

# =========================
# Run Application
# =========================
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )