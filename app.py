# app.py
import streamlit as st
import os
import pickle
from PIL import Image
import numpy as np
from feature_extractor import FaceEmbeddingExtractor


@st.cache_resource
def get_extractor():
    return FaceEmbeddingExtractor()


extractor = get_extractor()


@st.cache_data
def load_database():
    if os.path.exists("filenames.pkl") and os.path.exists("embeddings.pkl"):
        with open("filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return filenames, embeddings
    return None, None


filenames, embeddings = load_database()

st.set_page_config(layout="wide")
st.title("Bollywood Celeb lookalike")

if filenames is None or embeddings is None:
    st.error("Database files (filenames.pkl / embeddings.pkl) are missing. Please run feature_extractor.py first!")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Your Portrait")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)

    if uploaded_file is not None:
        with col2:
            st.header("Closest Twin Found")
            with st.spinner("Extracting face features and scanning database"):
                input_embedding = extractor.extract_embedding(input_image)

                if input_embedding is None:
                    st.error(
                        "Face not detected. Upload an image showing clear face")
                else:
                    scores = []
                    for db_emb in embeddings:
                        score = extractor.calculate_similarity(input_embedding, db_emb)
                        scores.append(score)

                    best_match_idx = int(np.argmax(scores))
                    best_score = scores[best_match_idx]
                    best_match_path = filenames[best_match_idx]

                    celeb_name = os.path.basename(os.path.dirname(best_match_path)).replace("_", " ").title()
                    if not celeb_name or celeb_name == ".":
                        celeb_name = os.path.splitext(os.path.basename(best_match_path))[0].replace("_", " ").title()

                    if os.path.exists(best_match_path):
                        st.image(Image.open(best_match_path), caption=f"Lookalike: {celeb_name}", use_container_width=True)
                        st.metric(label="Match Confidence", value=f"{best_score * 100:.2f}%")
                    else:
                        st.error(f"Match not found: {best_match_path}")