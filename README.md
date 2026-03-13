# 🧠 System Architecture
# Architecture Diagram

![Architecture
-Diagram
](architecture_diagram.png)

The application follows a multi-stage AI pipeline:

1. **Image Input**
   - User uploads or captures an architectural image

2. **Preprocessing**
   - Image resizing
   - Normalization
   - Noise reduction

3. **Level 1 Model**
   - CNN classifier
   - Predicts:
        Church
        Mosque
        Temple

4. **Level 2 Model**
   - Activated only if prediction = Temple
   - Classifies:
        Dravidian
        Nagara

5. **Prediction Layer**
   - Softmax probabilities
   - Confidence score visualization

6. **User Interface**
   - Streamlit dashboard
   - Prediction explanation
  
# 🧪 Model Training

## Data Preprocessing

The images were preprocessed using:

- Image resizing: 224 × 224
- Pixel normalization
- Data augmentation

Techniques used:

- Rotation
- Horizontal flip
- Zoom
- Brightness adjustment

## Training Details

Model: Convolutional Neural Network (CNN)

Training parameters:

Epochs: 20–50  
Batch Size: 32  
Optimizer: Adam  
Loss Function: Categorical Crossentropy  

## Evaluation Metrics

Accuracy  
Precision  
Recall  
F1 Score

# 🧠 Model Architecture

The system uses a Convolutional Neural Network (CNN) for image classification.

## pipeline

Input Image

    ↓

Image Preprocessing

    ↓

Convolution Layers
    
    ↓

Flatten Layer
    
    ↓

Dense Layer

    ↓

Softmax Output
    
    ↓

Prediction


# 📊 Model Performance

| Model | Accuracy |
|------|------|
| Architecture Classifier | 92% |
| Temple Style Classifier | 89% |

The model performs well in identifying architectural features such as:

- Towers
- Domes
- Sculptures
- Structural symmetry
  
# 🖼️ Application Demo

### Church Detection

![Church Prediction](chruch.png)

---

### Dravidian Temple Detection

![Dravidian Prediction](temple_dravidian%20.png)

---

### Nagara Temple Detection

![Nagara Prediction](temple_nagara.png)

# Requirements

## Create requirements.txt

streamlit

tensorflow

keras

opencv

python

numpy

pillow

matplotlib

scikit-learn

## Install dependencies:

pip install -r requirements.txt

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![License](https://img.shields.io/badge/License-MIT-green)

# 📂 Dataset 

Images were collected from open-source architectural datasets and manually curated images of:

• Churches
• Dravidian temples
• Nagara temples

link :- https://drive.google.com/drive/folders/1LbzF0nsc0NqjqXjtO9bMaDa-Hy9BqfNk?usp=sharing

# ⚙ Installation

Clone the repository

1. git clone https://github.com/manojrameshdev/Architectural_classification_with_Ai.git

2. cd Architectural_classification_with_Ai

3. Install dependencies

4. pip install -r requirements.txt

5. Run the application

6. streamlit run app.py

# 🛠 Tech Stack

Python

TensorFlow / Keras

OpenCV

Streamlit

NumPy

Scikit-learn

# 🚀 Future Improvements

1. Add more architectural styles

2. Improve dataset size

3. Deploy model online

4. Add Gen-Ai to re-construct the destroyed parts of architecture.

5. Improve model accuracy

# 👨‍💻 Author

Manoj Ramesh

## GitHub:
https://github.com/manojrameshdev
