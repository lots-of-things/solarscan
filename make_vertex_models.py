import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import torch.nn.functional as F
import json
import os
import subprocess

# Create a traceable model with preprocessing and inference pipeline
with open('ts_ex.json', 'r') as f:
    base64_image_string = json.load(f)['data']  # Example input size for ResNet


image_data = base64.b64decode(base64_image_string)
img = Image.open(BytesIO(image_data))
img = img.convert("RGB")
img = img.resize((200, 200))
image_tensor = transforms.ToTensor()(img)  # Convert to tensor
# Normalize the image and apply transformations
image_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])(image_tensor)

# Example pre-trained model (can be replaced with your own)
models_dir = 'models'
os.makedirs(f"{models_dir}/vertex_archives", exist_ok=True)
models_name = [m.split('_')[1][:-4] for m in os.listdir(models_dir) if m.endswith('.pth')]
for m in models_name:
    pretrained_model = torch.load(os.path.join(models_dir, "model_{}.pth".format(m)), weights_only=False, map_location=torch.device('cpu')) 
    with torch.no_grad():
        pretrained_model.eval()
        res = F.softmax(pretrained_model(image_tensor.unsqueeze(0)),dim=1).detach().cpu().numpy()[0,1]
    print(res)
    traced_model = torch.jit.trace(pretrained_model, image_tensor.unsqueeze(0))

    # # Save the traced model
    traced_model.save(f"{models_dir}/vertex_archives/model_{m}.pt")
    
    # Check that the traced model works just as well
    traced_model_load = torch.jit.load(f"{models_dir}/vertex_archives/model_{m}.pt")

    with torch.no_grad():
        traced_model_load.eval()
        res = F.softmax(traced_model_load(image_tensor.unsqueeze(0)),dim=1).detach().cpu().numpy()[0,1]
    print(res)
    
    # generate .mar for Vertex AI
    os.makedirs(f"{models_dir}/vertex_archives/model_{m}/", exist_ok=True)
    command = [
        "torch-model-archiver", 
        "--model-name", f"model",
        "--version", "1.0",
        "--serialized-file", f"{models_dir}/vertex_archives/model_{m}.pt",
        "--handler", f"{models_dir}/model_handler.py",
        "--export-path", f"{models_dir}/vertex_archives/model_{m}/",
        "--requirements-file", f"{models_dir}/requirements.txt",
        "--force"
    ]
    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        os.remove(f"{models_dir}/vertex_archives/model_{m}.pt")
        print("Model archived successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error while running torch-model-archiver: {e}")