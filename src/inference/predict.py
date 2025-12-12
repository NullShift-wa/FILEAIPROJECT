# code by : TranPhuocPhong
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
JSON_PATH  = os.path.join(BASE_DIR, "Assets", "class_indices.json")


model = tf.keras.models.load_model(MODEL_PATH)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    inv = {v: k for k, v in json.load(f).items()}
CLASS_NAMES = [inv[i] for i in sorted(inv.keys())]

def transform(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return CLASS_NAMES[idx], conf
