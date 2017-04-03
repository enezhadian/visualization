# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


def as_image(array, fmt='jpeg', s=0.1, size=None):
    array = np.atleast_2d((array - array.mean()) / max(array.std(), 1e-4) * s + 0.5)
    array = np.uint8(np.clip(array, 0, 1) * 255)
    image = Image.fromarray(array)
    if size is not None:
        image = image.resize(size, Image.NEAREST)
    return image
