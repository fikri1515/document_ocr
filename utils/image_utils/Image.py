import cv2 as cv
import numpy as np
from PIL.Image import Image
from PIL import Image as ImageMain

class Image_preprocessing:
    def __init__(self, image, scale=1):
        self._image = image

        self._default_cvImage = self.scale(self._image, scaling_factor=scale)

        # auto preprocessing image
        self._thresh = self._preprocessing_image(self._image, scale)

        # original cv image = original image that could be never change
        # binary image is an image that return from auto preprocessing function in init line code
        # cvImage image is an image that return from manual preprocessing function function code line

    def _load_image(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"image at path {image_path}, could not be loaded.")
        
        return image

    def _preprocessing_image(self, image, scale):
        image = self.remove_noise(image)
        image = self.scale(image, scaling_factor=scale)
        image = self.grayscale(image)
        image = self.thresholding(image)
        image = self.skel(image)
        image = self.normalization(image)
        # image, angle = self.deskew(image)

        return image
    
    # auto preprocessing can be using of default value function in beloew lines
    # skewing the image

    # noise removal function
    def remove_noise(self, image):
        return cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
    # image scaling function
    def scale(self, image, scaling_factor):
        height, width = image.shape[:2]

        new_dimension = (int(width * scaling_factor), int(height * scaling_factor))

        return cv.resize(image, new_dimension, interpolation=cv.INTER_LINEAR)

    # change image to grayscale
    def grayscale(self, image):

        blurred = cv.GaussianBlur(image, (5, 5), 0)
        return cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    # thresholding and binarizatio
    def thresholding(self, gray):
        # return cv.threshold(gray, 0, 255, cv.ADAPTIVE_THRESH_MEAN_C + cv.THRESH_OTSU)[1]

        # return cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)
        # ret, thresh = cv.threshold(image, 130, 255, cv.THRESH_BINARY_INV)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        if np.mean(thresh) > 127:
            thresh = cv.bitwise_not(thresh)

        return thresh

    # thinning and skeletonization
    def skel(self, image, kernel_points = 1):
        kernel = np.ones((kernel_points,kernel_points), np.uint8)
        
        return cv.erode(image, kernel, iterations=1)

    # normalize the image
    def normalization(self, image):
        norm_image = np.zeros((image.shape[0], image.shape[1]))
        return cv.normalize(image, norm_image, 0, 255, cv.NORM_MINMAX)
    
    def deskew(self, cvImage):
        if len(cvImage.shape) == 3:
            gray = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)
            gray = cv.bitwise_not(gray)

            # thresholding
            thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        elif len(cvImage.shape) == 2:
            thresh = cvImage
            
        coords = np.column_stack(np.where(thresh>0))
        angle = cv.minAreaRect(coords)[-1]

        if angle < -45:
            angle = - (90 + angle)
        else:
            angle = -angle


        # rotate cvImage to deskew it
        (h, w) = cvImage.shape[:2]
        center = (w // 2, h // 2)

        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(thresh, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

        return rotated, angle

def simple_threshold(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)
    
    return thresh

def image_padding(image, padding_percentage=0.2, image_size=(28, 28), GBR=False, interpolation=0):
    h, w = image.shape
    max_val = int(max(h, w))

    # padding precentage
    percentage_val = int(max_val * padding_percentage)

    # final padding value
    vert_val = (max_val - h) // 2
    hori_val = (max_val - w) // 2

    # additional padding
    additional_vert_padd = max_val - h - vert_val 
    additional_hori_padd = max_val - w - hori_val

    padded_image = cv.copyMakeBorder(
        image,
        vert_val + percentage_val,
        additional_vert_padd + percentage_val,
        hori_val + percentage_val,
        additional_hori_padd + percentage_val,
        borderType=cv.BORDER_CONSTANT,
        value=[0, 0, 0]  # adding [0,0,0] for padding, conditional with interpolation
    )

    resized_image = cv.resize(padded_image, (image_size), interpolation=interpolation)

    # return resized_image
    if GBR:
        # return image with 3 channel
        return cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)
    
    else:
        # return image with 1 channel
        return np.expand_dims(resized_image, axis=-1)

def showImage(cvImage) -> Image:
    return ImageMain.fromarray(cv.cvtColor(cvImage, cv.COLOR_BGR2RGB))