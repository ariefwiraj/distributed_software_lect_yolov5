from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import logging
import requests

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Endpoint API deteksi objek
FRUIT_DETECTION_URL = "http://192.168.1.8:5000/detect"
VEHICLE_DETECTION_URL = "http://192.168.1.6:5002/detect"
ELEKTRONIK_DETECTION_URL = "http://192.168.1.8:5003/detect"
WAJAH_DETECTION_URL = "https://8ec2-180-252-174-27.ngrok-free.app/predict"

# Halaman HTML dalam string
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Gateway Endpoint</title>
</head>
<body>
    <h1>Test Gateway Endpoint</h1>
    <form id="uploadForm" enctype="multipart/form-data">
        <label for="image">Upload an image:</label><br>
        <input type="file" id="image" name="image" accept="image/*" required><br><br>
        <button type="button" onclick="sendRequest()">Submit</button>
    </form>

    <h2>Response:</h2>
    <pre id="response">Response will appear here</pre>

    <script>
        async function sendRequest() {
            const form = document.getElementById('uploadForm');
            const formData = new FormData(form);

            const responseElement = document.getElementById('response');
            responseElement.textContent = 'Sending request...';

            try {
                const response = await fetch('/gateway', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                responseElement.textContent = JSON.stringify(result, null, 2);
            } catch (error) {
                responseElement.textContent = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/gateway', methods=['POST'])
def gateway():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']

    # Validasi file gambar
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

    # Deteksi buah
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

    # Deteksi kendaraan
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

    # Deteksi elektronik
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

    # Deteksi wajah
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)