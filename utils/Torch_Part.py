import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from RealESRGAN import RealESRGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tensor_transform = transforms.Compose([transforms.ToTensor()])

def realgan_loader(scale, device = device):
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    return model

def pil_to_tensor(image):
    return tensor_transform(image).unsqueeze(0).to(device)

# path_to_image = 'inputs/lr_image.png'
# image = Image.open(path_to_image).convert('RGB')

# sr_image = model.predict(image)

# sr_image.save('results/sr_image.png')