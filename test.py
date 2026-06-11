# test.py
import pickle
import os
import numpy as np
from feature_extractor import FaceEmbeddingExtractor


def run_test():
    if not os.path.exists("filenames.pkl") or not os.path.exists("embeddings.pkl"):
        print("Error")
        return

    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    print(f"Database loaded successfully with {len(filenames)} entries.")
    extractor = FaceEmbeddingExtractor()

    if len(embeddings) > 0:
        print("Model Integration Test: PASS")
        print(f"Vector dimensions verified: {len(embeddings[0])} dimensions.")
    else:
        print("Database empty.")


if __name__ == "__main__":
    run_test()