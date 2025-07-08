from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging

from model import UNET

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'saved_images'
ALLOWED_EXTENSIONS = {'jpg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoint.pth.tar"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

model = None

def load_model():
    global model
    try:
        model = UNET(in_ch=3, out_ch=1).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file {MODEL_PATH} not found. Using untrained model.")
            model.eval()
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask_tensor, original_size):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode='L')
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    return mask_image

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        try:
            image = Image.open(file.stream)
            original_size = image.size
            logger.info(f"Processing image of size: {original_size}")
            image_tensor = preprocess_image(image).to(DEVICE)
            with torch.no_grad():
                prediction = model(image_tensor)
                prediction = torch.sigmoid(prediction)
            mask_image = postprocess_mask(prediction, original_size)
            mask_base64 = image_to_base64(mask_image)
            filename = secure_filename(file.filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"segmented_{filename}")
            mask_image.save(output_path)
            return jsonify({
                'success': True,
                'result': mask_base64,
                'original_size': original_size,
                'processed_size': [IMAGE_WIDTH, IMAGE_HEIGHT]
            })
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return jsonify({
            'model_type': 'U-Net',
            'device': str(DEVICE),
            'input_channels': 3,
            'output_channels': 1,
            'image_height': IMAGE_HEIGHT,
            'image_width': IMAGE_WIDTH,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_path': MODEL_PATH,
            'model_loaded': os.path.exists(MODEL_PATH)
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Device: {DEVICE}")
    if load_model():
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model loading failed - server will run with untrained model")
    app.run(debug=True, host='0.0.0.0', port=5050)
