import albumentations as A
import numpy as np
import cv2


class Augmentations:
    def __init__(self):
        self.aug = A.Compose([
            A.RandomGamma(gamma_limit=(60, 160), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=1)
        ], p=1)
        pass

    def do(self, original):
        img = self.aug(image=original)['image']
        return img


