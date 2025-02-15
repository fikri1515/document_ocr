import cv2 as cv
import numpy as np

class char_segmentation:
    def __init__(self, cvImage, method='cca_char_segmentation') -> None:

        norm_cvImage = self.image_normalization(cvImage)

        if method == 'cca_char_segmentation':
            char_data, thresh, coords = self.cca_char_segmentation(cvImage)
        elif method == 'contour_char_segmentation':
            char_data, thresh, coords = self.contour_char_segmentation(cvImage)

        self._transformed_image = thresh
        
        self._cvImage = cvImage
        # self._norm_cvImage = norm_cvImage
        self._bound_image = char_data
        self._coords = coords

    def image_normalization(self, cvImage, height_normalization=50):
        desired_height = height_normalization
        height, width = cvImage.shape

        aspect_ratio = width / height
        new_width = int(desired_height * aspect_ratio)

        resized_image = cv.resize(cvImage, (new_width, desired_height))

        return resized_image
    
    def cca_char_segmentation(self, cvImage):
        # normalization dimension parameter
        rgb_resized_image = cv.cvtColor(cvImage, cv.COLOR_GRAY2BGR)

        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))

        # gunakan program untuk iterasi untuk menyesuaikan dengan lebar gambar atau menormalisasi gambar untuk mendapatkan height yang konsisten
        thresh = cv.morphologyEx(cvImage, cv.MORPH_CLOSE, rect_kernel, iterations=4)

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)

        coords = []
        for i in range(1, num_labels):
            x, y, w, h, area_char = stats[i]
            cv.rectangle(rgb_resized_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            coords.append([[x, y + h], [x, y], [x + w, y], [x + w, y + h]])

        coords = np.array(coords)

        return rgb_resized_image, thresh, coords
    
    def contour_char_segmentation(self, cvImage, kernel = (1, 3)):

        rgb_resized_image = cv.cvtColor(cvImage, cv.COLOR_GRAY2BGR)

        # example of using custom kernel
        # kernel_size = max(3, int(min(height, width) * 0.01))
        # kernel_size2 = max(3, int(max(height, width) * 0.17))

        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel)
        thresh = cv.morphologyEx(cvImage, cv.MORPH_CLOSE, rect_kernel, iterations=4)

        Contours, Hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        cv.drawContours(rgb_resized_image, Contours, -1, (0, 255, 0), 1)

        coords = []

        for contour in Contours:
            [x, y, w, h] = cv.boundingRect(contour)
            cv.rectangle(rgb_resized_image, (x, y), (x + w, y + h), (0,0,255), 1)
            coords.append([[x, y + h], [x, y], [x + w, y], [x + w, y + h]])

        coords = np.array(coords)

        return rgb_resized_image, thresh, coords