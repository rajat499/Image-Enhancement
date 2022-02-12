import numpy as np
from PIL import Image

def read_image(file):
    return np.array(Image.open(file))