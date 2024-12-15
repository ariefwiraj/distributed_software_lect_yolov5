import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

# Inisialisasi Flask
app = Flask(__name__)

# Load model YOLOv5
MODEL_PATH = "yolov5s.pt"  # Ganti dengan model YOLOv5 Anda yang sudah dilatih untuk mendeteksi buah
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Daftar kelas target (buah)
TARGET_CLASSES = ['apple', 'banana', 'orange', 'grape', 'pineapple']  # Sesuaikan dengan kelas yang ada pada model Anda

@app.route('/')
def home():
    return jsonify({"message": "YOLOv5 Fruit Detection API is running!"})

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # Ambil file gambar
    image_file = request.files['image']
    image = Image.open(BytesIO(image_file.read())).convert('RGB')

    # Deteksi objek
    results = model(image)

    # Filter hasil untuk kelas target
    detections = []
    for *box, conf, cls in results.xyxy[0].tolist():
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES and conf > 0.5:  # Confidence > 50%
            detections.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                }
            })

    return jsonify({"detections": detections})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
