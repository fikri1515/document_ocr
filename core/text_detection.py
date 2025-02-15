import cv2 as cv
import numpy as np

# new class for text detecting process
class DB_text_detection():
    def __init__(self, image, path_to_model, padding=False) -> None:
        if padding is not False:
            if padding == 'precentage':
                h, w = image.shape
                padding_x = int(0.4  * w)
                padding_y = int(0.3 * w)

                self._cvImage = cv.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv.BORDER_CONSTANT, value=0)
            elif type(padding) == int:
                padding_x = padding
                padding_y = padding

                self._cvImage = cv.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv.BORDER_CONSTANT, value=0)
        else:
            self._cvImage = image

        thresh_data, boxes, confidence = self.text_detection(self._cvImage, model_path=path_to_model)

        self._bound_image = thresh_data
        self._boxes = np.array(boxes)
        self._confidence = confidence

    def text_detection(self, image, model_path):
        model = cv.dnn.TextDetectionModel_DB(model_path)
        binThresh = 0.3
        polyThresh = 0.5
        maxCandidates = 200
        unclipRatio = 2.0

        model.setBinaryThreshold(binThresh)
        model.setPolygonThreshold(polyThresh)
        model.setMaxCandidates(maxCandidates)
        model.setUnclipRatio(unclipRatio)

        # normalization parameters
        scale = 1.0 / 255.0
        mean = (122.67891434, 116.66876762, 104.00698793)

        # The input shape
        inputSize = (736, 736)

        model.setInputParams(scale, inputSize, mean)

        # newimage = img.origcvimage
        # binimage = img.cvimage

        tchannel = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        boxes, confidence = model.detect(tchannel)

        for box in boxes:
            cv.polylines(tchannel, [box.astype(np.int32)], True, (0, 255, 0), 2)

        return tchannel, boxes, confidence
    
class EAST_text_detection():
    def __init__(self, image, model_path) -> None:
        self.cvImage = image

        bound_image, boxes, confidence = self.text_detection(image, model_path)
        
        self._bound_image = bound_image
        self._boxes = np.array(boxes)
        self._confidences = confidence

    def text_detection(self, image, model_path):
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        model = cv.dnn.TextDetectionModel_EAST(model_path)

        confThreshold = 0.5
        nmsThreshold = 0.4

        model.setConfidenceThreshold(confThreshold)
        model.setNMSThreshold(nmsThreshold)

        detScale = 1.0
        detInputSize = (320, 320)
        detMean = (123.68, 116.78, 103.94)
        swapRB = True

        model.setInputParams(detScale, detInputSize, detMean, swapRB)

        boxes, confidence = model.detect(image)
        cv.polylines(image, boxes, True, (0, 255, 0), 2)

        return image, boxes, confidence