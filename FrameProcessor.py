import cv2
import numpy as np


class FrameProcessor:

    def __init__(self):
        pass

    def blur(self, frame):
        # TODO looks better then gaussian blurring on images, check again when incoporating frames
        return cv2.medianBlur(frame, 5)

    def binarize(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(grayscale_frame, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     11, 2)  # without INV for white background

    def extract_grid(self, frame):
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        # frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        # print(contours)
        return frame

    def morph_open(self, frame):
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3,3)))


frame = cv2.imread('sudoku.jpeg')
cv2.imshow('frame', frame)
fp = FrameProcessor()
opened = fp.morph_open((fp.binarize(fp.blur(frame))))
binary_frame = (fp.binarize(fp.blur(frame)))


# binary_frame = fp.extract_grid(fp.binarize(fp.blur(frame)))
cv2.imshow('binary', binary_frame)
cv2.imshow('opened', opened)

cv2.waitKey(0)
