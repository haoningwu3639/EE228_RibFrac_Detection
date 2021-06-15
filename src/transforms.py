import cv2
import nibabel.processing
import numpy as np


class Resample:
    # Resample pre-processing
    def __init__(self, out_spacing):
        self.out_spacing = out_spacing
    
    def __call__(self, image):
        resampled_img = nibabel.processing.resample_to_output(image, self.out_spacing)
        return resampled_img


class Equalize:
    def __init__(self):
        self.firstflag = True
        self.startposition = 0
        self.endposition = 0

    def __call__(self, image):
        for z in range(image.shape[0]):
            notzeroflag = np.max(image[z])
            if notzeroflag and self.firstflag:
                self.startposition = z
                self.firstflag = False
            if notzeroflag:
                self.endposition = z
        assert (len(image.shape)==3)  #3D arrays
        #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_equalized = np.empty(image.shape)
        for i in range(self.startposition, self.endposition+1):
            imgs_equalized[i,:,:] = clahe.apply(np.array(image[i,:,:], dtype = np.uint8))
        return imgs_equalized


class Window:
    # The intensity of input voxels was clipped to the bone window (level=450, width=1100)
    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        image = np.clip(image, self.window_min, self.window_max)

        return image


class Normalize:
    # Normalized to [-1,1]
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        image = (image - self.low) / (self.high - self.low)
        image = image * 2 - 1

        return image
