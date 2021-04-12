import cv2
import numpy as np
from tensorflow.keras.models import load_model

MEDIAN_BLUR_KERNEL = 5
MIN_INTENSITY = 0
MAX_INTENSITY = 255
OPEN_KERNEL = np.ones((3, 3))
DILATE_KERNEL = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]).astype('uint8')


def process_image(img):
    # Convert image to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Median blur now that we have less channels
    blurred = cv2.medianBlur(grayscale, MEDIAN_BLUR_KERNEL)
    # Convert image to binary image using adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, MAX_INTENSITY,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   11, 2)  # INV for black background
    # Apply Opening to remove white noise spots
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, OPEN_KERNEL)
    return opened


def extract_board_corners(processed_img):
    copy = processed_img.copy()
    contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) < 10000:  # 10k for my webcam
            return None  # , cnt
        # accuracy parameter : maximum distance from contour to approximated contour
        epsilon = 0.1 * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)  # approximated curve
        if len(approx) == 4:  # found 4 corner of the sudoku grid
            corners = np.reshape(approx, (4, 2))
            return corners  # , cnt


def order_corners(corners):
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


def euclidean_distance(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_img_dimensions(corners):
    top_l, top_r, bottom_r, bottom_l = corners
    top_width = euclidean_distance(top_l, top_r)
    bottom_width = euclidean_distance(bottom_l, bottom_r)
    width = max(int(top_width), int(bottom_width))
    left_height = euclidean_distance(top_l, bottom_l)
    right_height = euclidean_distance(top_r, bottom_r)
    height = max(int(left_height), int(right_height))
    return height, width


def perspective_warp(image, corners):
    ordered_corners = order_corners(corners)
    height, width = get_img_dimensions(ordered_corners)
    # create new corners with the new dimensions ordered top_l, top_r, bottom_r, bottom_l
    new_corners = np.array([[0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
    # get transformation matrix and return the warped image
    M = cv2.getPerspectiveTransform(ordered_corners, new_corners)
    return cv2.warpPerspective(image, M, (width, height))


def is_empty(cell_img):
    # cell_img is of shape NxN, includes white borders of grid lines
    # true if at at least 90% of pixels are black (digits are white on black background)
    N = cell_img.shape[0]
    start = int(0.2 * N)
    center = cell_img[start: N - start, start: N - start]
    cell_img = center
    return cv2.countNonZero(cell_img) <= 0.10 * (cell_img.shape[0] * cell_img.shape[1])


def extract_digits(board_img):
    processed = process_image(board_img)
    # resize board to a square
    resized = cv2.resize(processed, (processed.shape[0], processed.shape[0]),
                         interpolation=cv2.INTER_AREA)
    pos_y = pos_x = resized.shape[0] // 9
    # grid_len = 10  # depends on the printed grid itself
    grid_zoom = 0.1  # 10%, depends mostly on grid
    board = np.zeros((9, 9))
    model = load_model('digits_cnn_v2.h5')
    for i in range(0, 9):
        for j in range(0, 9):
            # get the region of interest - remove white grid lines around cell
            # i.e., (grid_zoom)% surrounding pixel square around the image
            border = grid_zoom * i * pos_y
            top, bottom = i * pos_y + border , (i + 1) * pos_y - border
            left, right = j * pos_x + border, (j + 1) * pos_x - border
            cell = resized[top: bottom, left: right]
            if is_empty(cell):
                continue  # leave board[i][j] zero
            digit_img = cell
            # else, apply neural network to image, classify digit and insert to board
            digit_img = cv2.dilate(digit_img, DILATE_KERNEL)
            #     while True:
            #         cv2.imshow('current cell', cell)
            #         cv2.imshow('current digit', digit_img)
            #         if cv2.waitKey(1) & 0xFF == ord('q'):
            #             break
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = np.reshape(digit_img, (1, 28, 28, 1)).astype(np.float32) / 255.0
            board[i][j] = int(model.predict_classes(digit_img))
    return board


def extract_board(img):
    processed_orig = process_image(img)
    corners = extract_board_corners(processed_orig)
    if corners is None:
        return None
    board_img = perspective_warp(img, corners)
    board = extract_digits(board_img)
    return board


sudoku = cv2.imread('sudoku3.jpeg')
pred_board = extract_board(sudoku)
real_board = np.array([[9, 0, 0, 0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 5, 2, 0, 0, 0, 8],
                       [0, 0, 0, 7, 0, 0, 0, 0, 0],
                       [3, 0, 1, 0, 0, 9, 0, 0, 0],
                       [0, 0, 5, 3, 0, 8, 0, 0, 0],
                       [0, 7, 6, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 7, 4, 0],
                       [0, 2, 0, 0, 8, 0, 0, 6, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 5]])
compare = (pred_board == real_board)
# print(compare)
# print(pred_board)
