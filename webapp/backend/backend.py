from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from dotenv import load_dotenv
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import base64
from io import BytesIO

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)

# Load configuration values
app.config['VERSION'] = os.getenv('VERSION')
app.config['ENV'] = os.getenv('ENV')
if app.config['ENV'] == 'production':
    app.config['USE_CORS'] = os.getenv('USE_CORS').lower() == 'true'
    app.config['CORS_URL'] = os.getenv('CORS_URL')

    if app.config['USE_CORS']:
        CORS(app, origins=[app.config['CORS_URL']])

model_names = {
    "blurring",
    "spectral",
    "augmix",
    "standard",
    "oracle",
}

if app.config['ENV'] == 'production':
    from google.cloud import storage
    # Initialize the Google Cloud Storage client
    client = storage.Client()
    bucket_name = 'solarscan-models'  # Replace with your bucket name
    bucket = client.get_bucket(bucket_name)

    # Ensure the models directory in /tmp exists
    tmp_dir = '/tmp/models/'
    os.makedirs(tmp_dir, exist_ok=True)

    # Download model files from GCS to /tmp/models/
    models = {}
    for model_name in model_names:
        model_blob = bucket.blob(f'models/model_{model_name}.pth')
        model_path = os.path.join(tmp_dir, f'model_{model_name}.pth')
        model_blob.download_to_filename(model_path)
        
        # Load the model
        models[model_name] = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
        os.remove(model_path)
else:
    models = {model_name: torch.load(f'../../models/model_{model_name}.pth', weights_only=False, map_location=torch.device('cpu')) for model_name in model_names}

@app.route('/warmup', methods=['GET'])
def warmup():
    return {}

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json(force=True)
    model_name = request_data['model_name']
    base64_image_string = request_data['base64_image_string']
    
    image_data = base64.b64decode(base64_image_string)
    img = Image.open(BytesIO(image_data))
    img = img.convert("RGB")
    img = img.resize((200, 200))
    image_tensor = transforms.ToTensor()(img)  # Convert to tensor
    # Normalize the image and apply transformations
    image_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])(image_tensor)
    
    with torch.no_grad():
        models[model_name].eval()
        res = F.softmax(models[model_name](image_tensor.unsqueeze(0)), dim=1).detach().cpu().numpy()

    return jsonify(float(res[0,1]))

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=5050)

