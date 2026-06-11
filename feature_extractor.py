# feature_extractor.py
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pickle
import os
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class FaceEmbeddingExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=self.device
        )

        self.encoder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def extract_embedding(self, image_path_or_pil):
        try:
            if isinstance(image_path_or_pil, str):
                img = Image.open(image_path_or_pil).convert('RGB')
            else:
                img = image_path_or_pil.convert('RGB')
        except Exception:
            return None

        face_tensor = self.mtcnn(img)

        if face_tensor is None:
            return None

        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.encoder(face_tensor)

        return embedding.squeeze(0).cpu().numpy()

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


if __name__ == "__main__":
    extractor = FaceEmbeddingExtractor()

    if os.path.exists("filenames.pkl"):
        with open("filenames.pkl", "rb") as f:
            filenames = pickle.load(f)

        new_embeddings = []

        for idx, path in enumerate(filenames):
            print(f"[{idx + 1}/{len(filenames)}] Processing: {path}")
            embedding = extractor.extract_embedding(path)
            if embedding is not None:
                new_embeddings.append(embedding)
            else:
                print(f"Face not found in: {path}. Adding blank vector.")
                new_embeddings.append(np.zeros(512))

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(new_embeddings, f)
        print("Success")
    else:
        print("Error")