import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import base64
from io import BytesIO

class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if not isinstance(data, list):
            data = [data]  # Convert to list if it's not already a list
        preds = []
        for value in data:
            while isinstance(value, dict):
                value = value[next(iter(value))]
            image_data = base64.b64decode(value)
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
                self.model.eval()
                res = F.softmax(self.model(image_tensor.unsqueeze(0)), dim=1).detach().cpu().numpy()
            preds.append(float(res[0,1]))
        return preds
    
    