import cv2 as cv
import numpy as np

# new class for text detecting process
class text_detection():
    def __init__(self, image_path, path_to_model):
        super().__init__(image_path)
        
        thresh_data = self.text_detection(self._cvimage, model_path=path_to_model)

        self._bound_image = thresh_data['bin']
        self._boxes = np.array(thresh_data['boxes'])
        self._confidence = thresh_data['confidence']

        # cropped image from bounding boxes result
        self._cropped_image = None

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

        return {
            'bin' : tchannel,
            'boxes' : boxes,
            'confidence' : confidence
        }
    
    def sorting():
        pass

    def crop_function(self, image, coords):
        coords = np.array(coords, dtype=np.float32)

        width = max(int(np.linalg.norm(coords[1] - coords[2])), int(np.linalg.norm(coords[0] - coords[3])))
        height = max(int(np.linalg.norm(coords[1] - coords[0])), int(np.linalg.norm(coords[2] - coords[3])))

        dstPoints = np.array([[0, height], [0, 0], [width, 0], [width, height] ], dtype=np.float32)
        matrix = cv.getPerspectiveTransform(coords, dstPoints)

        croppedImage = cv.warpPerspective(image, matrix, (width, height))

        return croppedImage