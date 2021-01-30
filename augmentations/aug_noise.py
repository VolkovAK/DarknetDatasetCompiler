import albumentations as A
import numpy as np
import cv2


class Augmentations:
    def __init__(self):
        self.name = 'noise'
        self.aug = A.Compose([
            A.IAAAdditiveGaussianNoise(scale=(1., 5.), p=0.8),
            A.GaussNoise(var_limit=(10.0, 300.0), p=0.8),
            A.MultiplicativeNoise(p=0.8),
        ], p=1)

    def do(self, original):
        img = self.aug(image=original)['image']
        return img


