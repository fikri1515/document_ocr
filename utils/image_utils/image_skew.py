# image skewing class

import cv2 as cv
from graphics import image

# This service contains core methods needed to deskew images
class image_skew:
    def boundingBoxes(self, cvImage):
        # Prep image, copy, convert to gray scale, blur, and threshold
        contours_image = cvImage.copy()
        gray = cv.cvtColor(contours_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (9, 9), 0)
        thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
        dilate = cv.dilate(thresh, kernel, iterations=5)

        # Find all contours
        contours, *hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv.contourArea, reverse = True)

        for c in contours:
            rect = cv.boundingRect(c)
            x, y, w, h = rect
            cv.rectangle(contours_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

        return [contours, contours_image]

    def getSkewAngle(self, cvImage) -> float:
        
        contours = self.boundingBoxes(cvImage)

        # Find largest contour and surround in min area box
        largestContour = contours[0][0]
        minAreaRect = cv.minAreaRect(largestContour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
            return -1.0 * angle
        
        elif angle > 45:
            angle = 90 - angle
            return angle
        
        return -1.0 * angle
    
    # As your page gets more complex you might want to look into more advanced angle calculations
        #
        # Maybe use the average angle of all contours.
        # allContourAngles = [cv2.minAreaRect(c)[-1] for c in contours]
        # angle = sum(allContourAngles) / len(allContourAngles)
        #
        # Maybe take the angle of the middle contour.
        # middleContour = contours[len(contours) // 2]
        # angle = cv2.minAreaRect(middleContour)[-1]
        #
        # Maybe average angle between largest, smallest and middle contours.
        # largestContour = contours[0]
        # middleContour = contours[len(contours) // 2]
        # smallestContour = contours[-1]
        # angle = sum([cv2.minAreaRect(largestContour)[-1], cv2.minAreaRect(middleContour)[-1], cv2.minAreaRect(smallestContour)[-1]]) / 3
        #
        # Experiment and find out what works best for your case.


    # Deskew image
    def deskew(self, cvImage):

        try:
            angle = self.getSkewAngle(cvImage)
            return image().rotateImage(cvImage, -1.0 * angle)
        
        except:
            print('Error handling : cv image must be in only just original color object, and not preprocessed image')