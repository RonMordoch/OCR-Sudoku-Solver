import cv2
import numpy as np


class ImageProcessor:
    MEDIAN_BLUR_KERNEL = 5
    MIN_INTENSITY = 0
    MAX_INTENSITY = 255
    OPEN_KERNEL = np.ones((3, 3))
    DILATE_KERNEL = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]]).astype('uint8')

    def blur(self, image):
        # TODO median looks better than gaussian, check again with real frames
        return cv2.medianBlur(image, ImageProcessor.MEDIAN_BLUR_KERNEL)

    def rgb_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize(self, image):
        return cv2.adaptiveThreshold(image, ImageProcessor.MAX_INTENSITY,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     11, 2)  # INV for black background

    def extract_grid(self, image):
        copy = image.copy()
        contours, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            # accuracy parameter : maximum distance from contour to approximated contour
            epsilon = 0.1 * cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)  # approximated curve
            if len(approx) == 4:  # found 4 corner of the sudoku grid as (y,x) points
                approx = np.reshape(approx, (4, 2))
                top_r, top_l, bottom_l, bottom_r = approx[0], approx[1], approx[2], approx[3]
                corners = [tuple(top_l), tuple(top_r), tuple(bottom_r), tuple(bottom_l)]
                return cnt, corners
                # return cnt
        return None

    def morph_open(self, image):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, ImageProcessor.OPEN_KERNEL)

    def morph_dilate(self, image):
        # Kernel should catch the grid cells and dilate the board
        # Digits are also dilated as result
        return cv2.dilate(image, ImageProcessor.DILATE_KERNEL)

    def process_image(self, image):
        # Convert image to grayscale
        grayscale = self.rgb_to_grayscale(image)
        # Apply Median blur now that we have less channels
        blurred = self.blur(grayscale)
        # Convert image to binary image using adaptive threshold
        binary = self.binarize(blurred)
        # Apply Opening to remove white noise spots and Dilation to enhance cells for detection
        morphed = self.morph_dilate(self.morph_open(binary))
        # morphed = self.morph_open(binary)
        processed = morphed
        return processed


img_processor = ImageProcessor()
sudoko = cv2.imread('sudoku.jpeg')
board = img_processor.process_image(sudoko)
cnt, corners = img_processor.extract_grid(board)
# print(corners)
# print(corners[0][1], corners[1][1])
# board = board[corners[0][1]:corners[1][1], :]
while True:
    cv2.imshow('board', board)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
