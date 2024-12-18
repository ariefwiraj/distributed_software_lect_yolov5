import os
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from flask_cors import CORS
import logging
import requests

app = Flask(__name__)
CORS(app)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Load model YOLOv5
MODEL_PATH = "Fruits.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Daftar kelas target (buah)
TARGET_CLASSES = [
    "cucumber", "apple", "kiwi", "banana", "orange", "coconut", "peach",
    "cherry", "pear", "pomegranate", "pineapple", "watermelon", "melon",
    "grape", "strawberry"
]

# Endpoint API deteksi objek lainnya
FRUIT_DETECTION_URL = "http://127.0.0.1:5007/detect"
VEHICLE_DETECTION_URL = "http://192.168.1.6:5002/detect"
ELEKTRONIK_DETECTION_URL = "http://192.168.1.8:5003/detect"
WAJAH_DETECTION_URL = "https://8ec2-180-252-174-27.ngrok-free.app/predict"

@app.route('/')
def home():
    return jsonify({"message": "YOLOv5 Fruit Detection API is running!"})

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # Ambil file gambar
    image_file = request.files['image']
    try:
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    # Deteksi objek menggunakan YOLOv5
    results = model(image)

    # Filter hasil untuk kelas target
    detections = []
    for *box, conf, cls in results.xyxy[0].tolist():
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES and conf > 0.5:  # Confidence > 50%
            detection = {
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                }
            }
            detections.append(detection)

    response = {
        "status": "success",
        "detections": detections,
        "count": len(detections)
    }

    return jsonify(response)

@app.route('/detection', methods=['POST'])
def detection_with_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # Ambil file gambar
    image_file = request.files['image']
    try:
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    # Ubah ukuran gambar agar lebih kecil
    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    results = model(image)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    for *box, conf, cls in results.xyxy[0].tolist():
        class_name = results.names[int(cls)]
        if class_name in TARGET_CLASSES and conf > 0.5:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), f"{class_name} {conf:.2f}", fill="red", font=font)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/jpeg")

@app.route('/gateway', methods=['POST'])
def gateway():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']

    try:
        img = Image.open(image)
        img.verify()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    image.seek(0)

    files = {'image': (image.filename, image.read(), image.mimetype)}

    combined_results = {
        "fruit_detection_endpoint": FRUIT_DETECTION_URL,
        "vehicle_detection_endpoint": VEHICLE_DETECTION_URL,
        "elektronik_detection_endpoint": ELEKTRONIK_DETECTION_URL,
        "wajah_detection_endpoint": WAJAH_DETECTION_URL,
        "fruit_detection": [],
        "vehicle_detection": [],
        "elektronik_detection": [],
        "wajah_detection": []
    }

    try:
        fruit_response = requests.post(FRUIT_DETECTION_URL, files=files)
        logging.debug(f"Fruit detection response: {fruit_response.text}")
        fruit_response.raise_for_status()
        fruit_data = fruit_response.json()
        combined_results["fruit_detection"] = fruit_data.get("detections", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to fruit detection API: {str(e)}")
        combined_results["fruit_detection_error"] = str(e)

    image.seek(0)

    try:
        vehicle_response = requests.post(VEHICLE_DETECTION_URL, files=files)
        logging.debug(f"Vehicle detection response: {vehicle_response.text}")
        vehicle_response.raise_for_status()
        vehicle_data = vehicle_response.json()
        combined_results["vehicle_detection"] = vehicle_data.get("detected_objects", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to vehicle detection API: {str(e)}")
        combined_results["vehicle_detection_error"] = str(e)

    image.seek(0)

    try:
        elektronik_response = requests.post(ELEKTRONIK_DETECTION_URL, files=files)
        logging.debug(f"Elektronik detection response: {elektronik_response.text}")
        elektronik_response.raise_for_status()
        elektronik_data = elektronik_response.json()
        combined_results["elektronik_detection"] = elektronik_data.get("detections", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to elektronik detection API: {str(e)}")
        combined_results["elektronik_detection_error"] = str(e)

    image.seek(0)

    try:
        wajah_response = requests.post(WAJAH_DETECTION_URL, files=files)
        logging.debug(f"Wajah detection response: {wajah_response.text}")
        wajah_response.raise_for_status()
        wajah_data = wajah_response.json()
        combined_results["wajah_detection"] = wajah_data.get("results", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to wajah detection API: {str(e)}")
        combined_results["wajah_detection_error"] = str(e)

    return jsonify(combined_results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
