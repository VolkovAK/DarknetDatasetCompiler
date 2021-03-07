import albumentations as A
import numpy as np
import cv2


class Augmentations:
    def __init__(self):
        pass

    def do(self, original):
        img = 255 - original
        return img


