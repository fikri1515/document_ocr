import numpy as np
import cv2 as cv
from PIL.Image import Image
from PIL import Image as ImageMain

# sorting coords function according to axis 'x' or 'y', and 'y' by default
def sort_boxes(boxes, confidences = None, sort_by='y', reverse=False):
    top_left_points = boxes[:, 0, :]

    if sort_by == 'y':
        sorted_indices = np.lexsort((top_left_points[:, 0], top_left_points[:, 1]))

    elif sort_by == 'x':
        sorted_indices = np.lexsort((top_left_points[:, 1], top_left_points[:, 0]))

    # reversing array code line
    if reverse:
        sorted_indices = sorted_indices[::-1]
    
    # conditional checking, if there is any confidences input
    if confidences is not None:
        return [boxes[sorted_indices], confidences[sorted_indices]]
    else:
        return boxes[sorted_indices]

# image cropped function
def crop_image(cvImage, coords):

    image_container = []

    for i in range(len(coords)):
        box = np.array(coords[i], dtype=np.float32)

        width = max(int(np.linalg.norm(box[1] - box[2])), int(np.linalg.norm(box[0] - box[3])))
        height = max(int(np.linalg.norm(box[1] - box[0])), int(np.linalg.norm(box[2] - box[3])))

        dstPoints = np.array([[0, height], [0, 0], [width, 0], [width, height] ], dtype=np.float32)
        matrix = cv.getPerspectiveTransform(box, dstPoints)

        croppedImage = cv.warpPerspective(cvImage, matrix, (width, height), cv.INTER_NEAREST)

        image_container.append(croppedImage)

    return image_container