import os
import torch
from flask import Flask, request, jsonify, send_file, render_template_string
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
VEHICLE_DETECTION_URL = "https://2b06-202-51-197-51.ngrok-free.app/recognize_vehicle"
ELEKTRONIK_DETECTION_URL = "http://192.168.1.8:5003/detect"
WAJAH_DETECTION_URL = "https://8ec2-180-252-174-27.ngrok-free.app/predict"

@app.route('/')
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Detection API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 0;
            }

            h1 {
                color: #333;
            }

            form {
                margin-bottom: 20px;
            }

            .result {
                margin-top: 20px;
            }

            .json-output {
                white-space: pre-wrap;
                background: #f4f4f4;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }

            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 20px;
            }

            .error {
                color: red;
                font-weight: bold;
                margin-top: 10px;
            }

            .hidden {
                display: none;
            }

            .button-container {
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1>API DETECTION</h1>

        <!-- First Image Upload Form (Detect) -->
        <h2>Upload Image for Fruits Detection API</h2>
        <form id="uploadFormDetect" enctype="multipart/form-data">
            <label for="imageDetect">Upload an image:</label>
            <input type="file" id="imageDetect" name="image" accept="image/*" required>
            <div class="button-container">
                <button type="submit">Detect</button>
            </div>
        </form>

        <div class="result">
            <h3>Fruits Detection API Results (JSON):</h3>
            <button id="toggleJsonDetectButton" class="hidden">Hide JSON</button>
            <div id="jsonOutputDetect" class="json-output hidden"></div>
        </div>

        <div>
            <h3>Fruits Detection API Output Image with Bounding Box:</h3>
            <button id="toggleButtonDetect" class="hidden">Show Image</button>
            <img id="outputImageDetect" class="hidden" alt="Upload an image to see results">
        </div>

        <div style="margin-bottom: 60px;"></div>


        <!-- Second Image Upload Form (Gateway) -->
        <h2>Upload Image for Gateway API</h2>
        <form id="uploadFormGateway" enctype="multipart/form-data">
            <label for="imageGateway">Upload an image:</label>
            <input type="file" id="imageGateway" name="image" accept="image/*" required>
            <div class="button-container">
                <button type="submit">Detect</button>
            </div>
        </form>

        <div class="result">
            <h3>Gateway API Results (JSON):</h3>
            <button id="toggleJsonGatewayButton" class="hidden">Hide JSON</button>
            <div id="jsonOutputGateway" class="json-output hidden"></div>
        </div>

        <div class="error" id="errorOutput"></div>

        <script>
            // Detect API form submission
            document.getElementById('uploadFormDetect').addEventListener('submit', async function (event) {
                event.preventDefault();

                const imageInput = document.getElementById('imageDetect');
                const errorOutput = document.getElementById('errorOutput');
                const jsonOutput = document.getElementById('jsonOutputDetect');
                const outputImage = document.getElementById('outputImageDetect');
                const toggleButton = document.getElementById('toggleButtonDetect');
                const toggleJsonButton = document.getElementById('toggleJsonDetectButton');

                // Reset previous results and errors
                errorOutput.textContent = '';
                jsonOutput.textContent = '';
                jsonOutput.classList.add('hidden');
                toggleJsonButton.classList.add('hidden');
                outputImage.src = '';
                outputImage.classList.add('hidden');
                toggleButton.classList.add('hidden');

                // Validate if an image is selected
                if (!imageInput.files.length) {
                    errorOutput.textContent = "Please upload an image.";
                    return;
                }

                const formData = new FormData();
                formData.append('image', imageInput.files[0]);

                try {
                    // Step 1: Fetch Detect API JSON results
                    const jsonResponse = await fetch('https://e016-2001-448a-2061-4a86-4151-25b4-9c43-37f0.ngrok-free.app/detect', {
                        method: 'POST',
                        body: formData
                    });

                    if (!jsonResponse.ok) {
                        throw new Error("Failed to fetch JSON response.");
                    }

                    const jsonData = await jsonResponse.json();
                    jsonOutput.textContent = JSON.stringify(jsonData, null, 2);
                    jsonOutput.classList.remove('hidden');
                    toggleJsonButton.classList.remove('hidden');
                    toggleJsonButton.textContent = 'Hide JSON';

                    // Step 2: Fetch output image with bounding boxes
                    const imageResponse = await fetch('https://e016-2001-448a-2061-4a86-4151-25b4-9c43-37f0.ngrok-free.app/detection', {
                        method: 'POST',
                        body: formData
                    });

                    if (!imageResponse.ok) {
                        throw new Error("Failed to fetch output image.");
                    }

                    const imageBlob = await imageResponse.blob();
                    const imageUrl = URL.createObjectURL(imageBlob);
                    outputImage.src = imageUrl;
                    outputImage.classList.remove('hidden');
                    toggleButton.classList.remove('hidden');
                    toggleButton.textContent = 'Hide Image';
                } catch (error) {
                    errorOutput.textContent = `An error occurred: ${error.message}`;
                    console.error("Error:", error);
                }
            });

            // Gateway API form submission
            document.getElementById('uploadFormGateway').addEventListener('submit', async function (event) {
                event.preventDefault();

                const imageInput = document.getElementById('imageGateway');
                const errorOutput = document.getElementById('errorOutput');
                const jsonOutput = document.getElementById('jsonOutputGateway');

                // Reset previous results and errors
                errorOutput.textContent = '';
                jsonOutput.textContent = '';
                jsonOutput.classList.add('hidden');
                const toggleJsonButton = document.getElementById('toggleJsonGatewayButton');
                toggleJsonButton.classList.add('hidden');

                // Validate if an image is selected
                if (!imageInput.files.length) {
                    errorOutput.textContent = "Please upload an image.";
                    return;
                }

                const formData = new FormData();
                formData.append('image', imageInput.files[0]);

                try {
                    // Fetch Gateway API JSON results
                    const jsonResponse = await fetch('https://e016-2001-448a-2061-4a86-4151-25b4-9c43-37f0.ngrok-free.app/gateway', {
                        method: 'POST',
                        body: formData
                    });

                    if (!jsonResponse.ok) {
                        throw new Error("Failed to fetch JSON response.");
                    }

                    const jsonData = await jsonResponse.json();
                    jsonOutput.textContent = JSON.stringify(jsonData, null, 2);
                    jsonOutput.classList.remove('hidden');
                    toggleJsonButton.classList.remove('hidden');
                    toggleJsonButton.textContent = 'Hide JSON';
                } catch (error) {
                    errorOutput.textContent = `An error occurred: ${error.message}`;
                    console.error("Error:", error);
                }
            });

            // Toggle JSON visibility
            function toggleJsonVisibility(jsonOutput, toggleJsonButton) {
                if (jsonOutput.classList.contains('hidden')) {
                    jsonOutput.classList.remove('hidden');
                    toggleJsonButton.textContent = 'Hide JSON';
                } else {
                    jsonOutput.classList.add('hidden');
                    toggleJsonButton.textContent = 'Show JSON';
                }
            }

            // Add event listeners for toggle buttons
            document.getElementById('toggleJsonDetectButton').addEventListener('click', function () {
                toggleJsonVisibility(document.getElementById('jsonOutputDetect'), document.getElementById('toggleJsonDetectButton'));
            });

            document.getElementById('toggleJsonGatewayButton').addEventListener('click', function () {
                toggleJsonVisibility(document.getElementById('jsonOutputGateway'), document.getElementById('toggleJsonGatewayButton'));
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

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
        # Tambahkan data lengkap dari vehicle_response
        combined_results["vehicle_detection"] = vehicle_data
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
