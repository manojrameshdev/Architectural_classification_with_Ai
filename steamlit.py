import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# --- Modern, stylish fonts and glowing effects ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Montserrat:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        background-color: #181927;
        color: #EEEEEE;
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #29FFE2;
        text-shadow: 0 0 7px #29ffe2, 0 0 24px #2e3a79;
    }
    .stApp { background: linear-gradient(135deg, #181927 65%, #2e3a79 100%);}
    div[data-testid="stMarkdownContainer"], .markdown-text-container {
        color: #e0e0e0;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.1em;
    }
    .stButton > button {
        background-color: #29FFE2;
        color: #181927;
        border-radius: 5px;
        font-weight: bold;
        box-shadow: 0 0 10px #29FFE2;
    }
    .stCaption, .css-139e0w6 { color: #A3FFCE; font-style: italic; }
    .stInfo { background-color: #2e3a79; color: #fcf96f; border-radius: 8px; }
    .stMarkdown { border-left: 4px solid #29FFE2; padding-left: 12px; }
    .stImage img { box-shadow: 0 0 10px #29FFE2; border-radius: 20px; }
    .css-1au5q1h { color: #29FFE2; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Architectural Classifier", page_icon="🏛️", layout="centered")

try:
    model1 = tf.keras.models.load_model('main_classifier.h5')
    model2 = tf.keras.models.load_model('temple_classifier.h5')
except Exception:
    st.error("Models not found. Train with app.py first.")

main_labels = {0: "Church", 1: "Mosque", 2: "Temple"}
temple_labels = {0: "dravidian", 1: "nagara"}

def preprocess_img(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    if len(arr.shape) == 2:  # grayscale photo
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4:  # RGBA image, drop alpha
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0)

st.title("🏛️ Architectural Image Classifier")
st.markdown("""
Upload or capture an image of an architectural building.<br>
<b style='color:#29FFE2'>The AI will predict its type and temple style with glowing confidence bars!</b>
""", unsafe_allow_html=True)

img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
camera_img = st.camera_input("Or capture one")

if img_file or camera_img:
    try:
        # Camera images and uploads can be handled identically if using PIL.Image.open(io.BytesIO(...))
        imgdata = img_file if img_file is not None else camera_img
        img = Image.open(imgdata)
        st.image(img, caption="Input Image", use_column_width=True)
        arr = preprocess_img(img)
        probs1 = model1.predict(arr)
        pred1 = np.argmax(probs1, axis=1)[0]
        label1 = main_labels[pred1]
        confidence1 = probs1[0][pred1]

        # Warn if confidence is low
        if max(probs1[0]) < 0.5:
            st.warning("This image may not depict a church, mosque, or temple. Please try a clearer or different photo.")

        st.markdown(f"<h3>Level 1 Classification: <span style='color:#fcf96f'>{label1}</span></h3>", unsafe_allow_html=True)
        st.markdown(f"<b>Confidence Score:</b>", unsafe_allow_html=True)
        st.progress(float(confidence1))
        st.write({k: f"{v:.2f}" for k,v in dict(zip(main_labels.values(), probs1[0])).items()})

        if label1 == "Temple":
            probs2 = model2.predict(arr)
            pred2 = np.argmax(probs2, axis=1)[0]
            label2_display = temple_labels[pred2].capitalize()
            confidence2 = probs2[0][pred2]
            st.markdown(f"<h3>Level 2 (Temple Style): <span style='color:#29FFE2'>{label2_display}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<b>Confidence Score:</b>", unsafe_allow_html=True)
            st.progress(float(confidence2))
            st.write({k.capitalize(): f"{v:.2f}" for k,v in dict(zip(temple_labels.values(), probs2[0])).items()})

            if label2_display == "Dravidian":
                st.info("🏯 <b>Dravidian:</b> Southern Indian temple style; tall gopurams, vibrant colors and sculptures.", icon="🏯")
            elif label2_display == "Nagara":
                st.info("🏰 <b>Nagara:</b> Northern Indian temple style; beehive-shaped towers (shikharas), simpler exterior.", icon="🏰")
        elif label1 == "Church":
            st.info("🕍 <b>Church:</b> Steeples, Gothic/Baroque or Modern styles; stained glass, cross iconography.", icon="🕍")
        elif label1 == "Mosque":
            st.info("🕌 <b>Mosque:</b> Domes, Minarets, Arches, and Islamic geometric ornaments.", icon="🕌")
    except Exception as e:
        st.error(f"Prediction or image reading error: {e}")
else:
    st.info("Please upload or capture an image to classify.")

st.markdown("---")
st.caption("✨ Research tool | Streamlit & TensorFlow | Made for visibility and clarity")
