import albumentations as A
import numpy as np
import cv2


class Augmentations:
    def __init__(self):
        self.aug = A.Compose([
            A.IAAPerspective(scale = (0.05, 0.2), p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, p=0.5),
            A.GridDistortion(distort_limit=0.5, p=0.5),
            A.ElasticTransform(p=0.5)
        ], p=1)
        pass

    def do(self, original):
        img = self.aug(image=original)['image']
        return img


