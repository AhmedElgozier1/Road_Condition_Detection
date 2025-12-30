# 🛣️ Road Condition Detection System

## 📌 Problem Definition
Poor road conditions such as potholes, cracks, and traffic congestion can lead to accidents, vehicle damage, and delays.
This project presents a **Deep Learning–based system** that analyzes road images and classifies road conditions to help drivers and road maintenance authorities.

---

## 📊 Data Collection
- Road images collected from publicly available datasets and online sources.
- The dataset contains multiple road condition categories:
  - Good Road
  - Potholes
  - Cracks
  - Traffic Congestion
- Dataset size: **Thousands of images** with different lighting and weather conditions.

---

## 🧹 Data Cleaning & Analysis
- Image resizing to a fixed input size
- Normalization of pixel values
- Removal of corrupted images
- Data augmentation (rotation, flipping, brightness changes)
- Splitting the dataset into training and testing sets

---

## 🧠 Feature Engineering
This project uses an **end-to-end Deep Learning approach**, where feature extraction is automatically learned by the convolutional layers of the neural network without manual feature engineering.

---

## 🏗️ Model Design
- Framework: **PyTorch**
- Model Type: **Convolutional Neural Network (CNN)**
- Architecture includes:
  - Convolutional layers
  - Pooling layers
  - Fully connected layers
- Output layer classifies the image into one of four road condition classes.

---

## 🏋️ Model Training
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Training conducted for multiple epochs
- Training implemented in Jupyter Notebook
- Model trained using CPU/GPU depending on availability

---

## 🧪 Model Testing & Inference
- Model evaluated on unseen test images
- Performance assessed using accuracy and qualitative evaluation
- Example predictions demonstrate correct classification behavior

---

## 🖥️ GUI Implementation
A graphical user interface was developed using **Gradio** to demonstrate the model in a real-world application scenario.

### GUI Features:
- Upload road images
- Run inference
- Display predicted road condition
- Show confidence scores for all classes

### 🔗 Live Demo
https://5e3a0232fe28e0df5d.gradio.live 

📸 GUI screenshots are available in the `screenshots/` folder.

> **Note:**  
> The current GUI uses a mock inference module for demonstration purposes.  
> The trained PyTorch model can be integrated easily by replacing the dummy prediction function.

---

## 👥 Team Contributions
This project was developed by a team of **7 members** with fair task distribution:

| Member | Contribution |
|------|-------------|
| Member 1 | Problem definition & dataset research |
| Member 2 | Data preprocessing & analysis |
| Member 3 | Feature engineering |
| Member 4 | Model architecture design |
| Member 5 | Model training & optimization |
| Member 6 | Testing & evaluation |
| Member 7 | GUI development & deployment |

---

## 🚀 How to Run
```bash
pip install gradio numpy pillow
python app.py
```

## Conclusion

This project demonstrates the application of Deep Learning techniques to solve a real-world problem related to road safety.  
It integrates model training, inference, and a user-friendly GUI, making it suitable for both academic and practical use.
