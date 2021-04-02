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

    ###### OPEN-CV WRAPPERS ######

    def blur(self, image):
        # median looks better than gaussian, check again with real frames
        return cv2.medianBlur(image, ImageProcessor.MEDIAN_BLUR_KERNEL)

    def rgb_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize(self, image):
        return cv2.adaptiveThreshold(image, ImageProcessor.MAX_INTENSITY,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     11, 2)  # INV for black background

    def morph_open(self, image):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, ImageProcessor.OPEN_KERNEL)

    def morph_dilate(self, image):
        # Kernel should catch the grid cells and dilate the board
        # Digits are also dilated as result
        return cv2.dilate(image, ImageProcessor.DILATE_KERNEL)

    ###### SUDOKU GRID EXTRACTION METHODS ######

    def get_board_corners(self, image):
        copy = image.copy()
        contours, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10000:
                return None
            # accuracy parameter : maximum distance from contour to approximated contour
            epsilon = 0.1 * cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)  # approximated curve
            if len(approx) == 4:  # found 4 corner of the sudoku grid
                corners = np.reshape(approx, (4, 2))
                return corners
        return None

    def euclidean_distance(self, p1, p2):
        return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def get_img_dimensions(self, corners):
        top_l, top_r, bottom_r, bottom_l = corners
        top_width = self.euclidean_distance(top_l, top_r)
        bottom_width = self.euclidean_distance(bottom_l, bottom_r)
        width = max(int(top_width), int(bottom_width))
        left_height = self.euclidean_distance(top_l, bottom_l)
        right_height = self.euclidean_distance(top_r, bottom_r)
        height = max(int(left_height), int(right_height))
        return height, width

    def order_corners(self, corners):
        # Order corner points in clockwise order: top-left, top-right, bottom-right, bottom-left
        ordered = np.zeros((4, 2), dtype=np.float32)
        pts_sum = np.sum(corners, axis=1)
        # top-left corner is the point (a,b) with minimum sum of a+b
        ordered[0] = corners[np.argmin(pts_sum)]
        # bottom-right corner is the point (a,b) with maximum sum of a+b
        ordered[2] = corners[np.argmax(pts_sum)]
        pts_diff = np.diff(corners, axis=1)
        # top-left corner is the point (a,b) with minimum diff of a-b (small x, large y)
        ordered[1] = corners[np.argmin(pts_diff)]
        # bottom-left corner is the point (a,b) with maximum diff of a-b (large x, small y)
        ordered[3] = corners[np.argmax(pts_diff)]
        return ordered

    def perspective_warp(self, image, corners):
        ordered_corners = self.order_corners(corners)
        height, width = self.get_img_dimensions(ordered_corners)
        # create new corners with the new dimensions ordered top_l, top_r, bottom_r, bottom_l
        new_corners = np.array([[0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1],
                                [0, height - 1]], dtype=np.float32)
        # get transformation matrix and return the warped image
        M = cv2.getPerspectiveTransform(ordered_corners, new_corners)
        return cv2.warpPerspective(image, M, (width, height))

    def process_image(self, image):
        # Convert image to grayscale
        grayscale = self.rgb_to_grayscale(image)
        # Apply Median blur now that we have less channels
        blurred = self.blur(grayscale)
        # Convert image to binary image using adaptive threshold
        binary = self.binarize(blurred)
        morphed = binary
        # Apply Opening to remove white noise spots and Dilation to enhance cells for detection
        morphed = self.morph_dilate(self.morph_open(binary))
        morphed = self.morph_open(binary) # TODO decide if necessary with real frames
        processed = morphed
        return processed


img_processor = ImageProcessor()
sudoku = cv2.imread('sudoku2.jpeg')
board = img_processor.process_image(sudoku)
corners = img_processor.get_board_corners(board)
warped = img_processor.perspective_warp(sudoku, corners)
warped2 = img_processor.perspective_warp(board, corners)
while True:
    cv2.imshow('board', board)
    cv2.imshow('warped', warped)
    cv2.imshow('warped2', warped2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
