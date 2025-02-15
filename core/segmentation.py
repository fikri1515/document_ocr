import cv2 as cv
import numpy as np

class word_segmentation:
    def __init__(self, image) -> None:
        self._image = image

        output_image, thresh, transformed_thresh, stats, cropped_image=self.cca_segmentation(image)
        self._stats = stats
        self._bound_image = output_image
        self._thresh = thresh
        self._transformed_thresh = transformed_thresh
        self._cropped_image = cropped_image

    def cca_segmentation(self, image):

        output_image = image.copy()

        blurred = cv.GaussianBlur(image, (5, 5), 0)
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        if np.mean(thresh) > 127:
            thresh = cv.bitwise_not(thresh)

        kernel = np.ones((5, 6), np.uint8)
        transformed_thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

        n_labels, labels, stats, _= cv.connectedComponentsWithStats(transformed_thresh, connectivity=8)

        stats = sorted(stats, key=lambda x: x[0])

        cropped_image = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            cv.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropped_image.append(image[y:y+h, x:x+w])
            
        return output_image, thresh, transformed_thresh, stats, cropped_image
    

class char_segmentation:
    def __init__(self, image) -> None:
        self._image =  image

        bound_image, letters, thresh, transformed_thresh, n_labels, labels, stats, masked_image = self.cca_segmentation(image)

        self._bound_image = bound_image
        self._cropped_image = letters
        self._thresh = thresh
        self._transformed_thresh = transformed_thresh
        self._nlabels = n_labels
        self._labels = labels
        self._stats = stats
        self._mask = masked_image

    def cca_segmentation(self, image, padding_precentage=0.2):
        output_image = image.copy()

        blurred = cv.GaussianBlur(image, (5, 5), 0)
        gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


        # need more developing
        if np.mean(thresh) > 127:
            thresh = cv.bitwise_not(thresh)

        h, w= thresh.shape
        max_val = int(max(h, w))
        padding_val = int(max_val * padding_precentage)

        thresh = cv.copyMakeBorder(
        thresh,
        padding_val,
        padding_val,
        padding_val,
        padding_val,
        borderType=cv.BORDER_CONSTANT,
        value=([0, 0, 0])  # adding [0,0,0] for padding, conditional with interpolation
        )

        # optionally adding dilation and erotion for connected characters, below this line
        kernel = np.ones((5, 1), np.uint8)

        transformed_thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

        n_labels, labels, stats, _= cv.connectedComponentsWithStats(transformed_thresh, connectivity=8)

        label_positions = np.argsort(stats[:, cv.CC_STAT_LEFT])
        sorted_stats = stats[label_positions]

        cropped_image = []
        masked_image = []

        for i in range(1, n_labels):
            # create bounding boxes for _bound_image
            x, y, w, h, area = sorted_stats[i]

            # masking line code
            mask = np.zeros_like(thresh)
            mask[labels == label_positions[i]] = 255
            result = cv.bitwise_and(thresh, thresh, mask=mask)

            # append cropped masking to list
            masked_image.append(result[y:y+h, x:x+w])

            x = x - padding_val
            y = y - padding_val
            cv.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cropped_image.append(image[y:y+h, x:x+w])

        return output_image, cropped_image, thresh, transformed_thresh, n_labels, labels, stats, masked_image