# Bollywood Celeb Lookalike Finder

A deep learning–powered web app that detects your face and finds your closest Bollywood celebrity twin from a pre-built image database — using facial embeddings and cosine similarity.

---

## Demo

<img width="1438" height="785" alt="image" src="https://github.com/user-attachments/assets/69836050-1fe3-46e9-8330-257b9a4c0f05" />

---

## How It Works

1. **Face Detection** — MTCNN detects and crops your face from the uploaded image.
2. **Feature Extraction** — InceptionResnetV1 (pretrained on VGGFace2) converts the face into a 512-dimensional embedding vector.
3. **Similarity Search** — Cosine similarity is computed between your embedding and every celebrity embedding in the database.
4. **Best Match** — The celebrity with the highest similarity score is returned as your lookalike.

---

## Project Structure

```
bollywood-twin-finder/
│
├── app.py                  # Streamlit web application
├── feature_extractor.py    # Face detection + embedding extraction
├── test.py                 # Sanity test for the database and model
│
├── filenames.pkl           # List of celebrity image paths (generated)
├── embeddings.pkl          # Pre-computed face embeddings (generated)
│
└── dataset/                # Your celebrity image dataset
    ├── Shah_Rukh_Khan/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── Deepika_Padukone/
    └── ...
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bollywood-twin-finder.git
cd bollywood-twin-finder
```

### 2. Install dependencies

```bash
pip install torch torchvision facenet-pytorch streamlit pillow numpy
```

> **GPU support (optional but recommended):** Install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Prepare your dataset

Organize celebrity images in the following folder structure:

```
dataset/
├── Celebrity_Name/
│   ├── image1.jpg
│   └── image2.jpg
└── Another_Celebrity/
    └── image1.jpg
```

Then generate `filenames.pkl` by running:

```python
import pickle, glob

filenames = glob.glob("dataset/**/*.jpg", recursive=True)
with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)
```

### 4. Build the embeddings database

```bash
python feature_extractor.py
```

This processes every image in `filenames.pkl`, extracts face embeddings, and saves them to `embeddings.pkl`. This step only needs to be run once (or whenever your dataset changes).

### 5. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Testing

Run the sanity test to verify the database loaded correctly and the model is working:

```bash
python test.py
```

Expected output:
```
Database loaded successfully with N entries.
Model Integration Test: PASS
Vector dimensions verified: 512 dimensions.
```

---

## Tech Stack

| Component | Library |
|---|---|
| Face Detection | [MTCNN](https://github.com/timesler/facenet-pytorch) |
| Face Recognition | [InceptionResnetV1 (VGGFace2)](https://github.com/timesler/facenet-pytorch) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Web UI | [Streamlit](https://streamlit.io/) |
| Image Processing | [Pillow](https://pillow.readthedocs.io/) |

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- A dataset of Bollywood celebrity images organized by name in subfolders

---

## Notes

- For best results, upload a **clear, front-facing portrait** with good lighting.
- The app detects **one face per image** (the most prominent one).
- If no face is detected, it will prompt you to try a different photo.
- Match confidence is based on **cosine similarity** — a score above 70% generally indicates a strong resemblance.

---

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Tim Esler for the MTCNN and InceptionResnetV1 implementations.
- [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) for the pretrained model weights.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
