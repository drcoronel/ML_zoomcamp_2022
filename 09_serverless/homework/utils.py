from io import BytesIO
from urllib import request
import numpy as np

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0

def process_from_url(url:str,target_size:tuple=(150,150)):
    img = download_image(url)
    img = prepare_image(img,target_size=target_size)
    x = np.array(img,dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    return X 



