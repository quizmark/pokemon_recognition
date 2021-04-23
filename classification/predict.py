import os
import time
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from PIL import Image
from torch import nn
from config import config
from model import get_models
from pokemons import load_data
from torchvision import transforms
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def img2str(image):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    _, buffer = cv2.imencode('.jpg', image, encode_params)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text.decode()

def str2img(uri):
    uri = base64.b64decode(uri.encode())
    nparr = np.frombuffer(uri, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join('{}{}'.format(config['modelroot'], config['num_used_classes']), config['best_model'], 'model_final.pt')
    model = torch.load(model_path)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    _, _, categories = load_data(config)

    return model, device, categories


model, device, categories = load_model(config)
print('model is loaded')

def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.NEAREST),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    with torch.no_grad():
        image = transform(image)
        image = image.to(device)
        output = model(image.unsqueeze(0))
    
    return F.softmax(output[0]).cpu().numpy()


def get_output(output):
    res_index = np.argmax(output)

    confident_score = output[res_index]
    class_name = categories[res_index]

    return class_name, confident_score

class Item_recognition(BaseModel):
    image: str

@app.post('/recognition')
async def recognition(item: Item_recognition):
    image = str2img(item.image)
    cv2.imwrite('input.png', image)
    result = get_output(predict(image))
    print(result)
    return {'name': result[0], 'score': float(result[1])}

if __name__ == "__main__":
    

    uvicorn.run("predict:app", host="0.0.0.0", port=5000, log_level="info", reload="true")



