import cv2
import numpy as np
from tensorflow.keras.models import load_model
import SudokuSolver

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

    copy = processed_img.copy()
    contours, _ = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20000:  # for my webcam
            return None
        # accuracy parameter : maximum distance from contour to approximated contour
        epsilon = 0.1 * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)  # approximated curve
        if len(approx) == 4:  # found 4 corner of the sudoku grid
            corners = np.reshape(approx, (4, 2))
            return order_corners(corners)


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
    height, width = get_img_dimensions(corners)
    # create new corners with the new dimensions ordered top_l, top_r, bottom_r, bottom_l
    new_corners = np.array([[0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
    # get transformation matrix and return the warped image
    M = cv2.getPerspectiveTransform(corners, new_corners)
    return cv2.warpPerspective(image, M, (width, height)), M


def is_empty(cell_img):
    # cell_img is of shape NxN, includes white borders of grid lines
    # true if at at least 90% of pixels are black (digits are white on black background)
    N = cell_img.shape[0]
    start = int(0.2 * N)
    center = cell_img[start: N - start, start: N - start]
    cell_img = center
    return cv2.countNonZero(cell_img) <= 0.10 * (cell_img.shape[0] * cell_img.shape[1])


def extract_board(board_img):
    processed = process_image(board_img)
    # resize board to a square
    resized = cv2.resize(processed, (processed.shape[0], processed.shape[0]),
                         interpolation=cv2.INTER_AREA)
    pos_y = pos_x = resized.shape[0] // 9
    grid_len = 10  # depends on the printed grid itself
    board = np.zeros((9, 9)).astype(np.int)
    model = load_model('digits_cnn_v2.h5')
    for i in range(0, 9):
        for j in range(0, 9):
            # get the region of interest - remove white grid lines around cell
            top, bottom = i * pos_y + grid_len, (i + 1) * pos_y - grid_len
            left, right = j * pos_x + grid_len, (j + 1) * pos_x - grid_len
            cell = resized[top: bottom, left: right]
            if is_empty(cell):
                continue  # leave board[i][j] zero
            digit_img = cell
            # else, apply neural network to image, classify digit and insert to board
            # while True: TODO debug delete
            #     cv2.imshow('current digit', digit_img)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # cv2.destroyAllWindows()
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = np.reshape(digit_img, (1, 28, 28, 1)).astype(np.float32) / 255.0
            board[i][j] = int(model.predict_classes(digit_img))
    return board


def fill_solution(board_img, grid_unsolved, grid_solved):
    height, width = board_img.shape[0], board_img.shape[1]  # warped board img dimensions
    # pos_y is the bottom side , pos_x is the left side
    pos_y, pos_x = height // 9, width // 9
    offset_y, offset_x = int(0.03 * height), int(0.03 * width)
    for i in range(9):
        for j in range(9):
            if grid_unsolved[i][j] == 0:
                digit = grid_solved[i][j]
                x = pos_x * j + offset_x  # add offset to position digit to center
                y = pos_y * (i + 1) - offset_y  # decrease offset to raise digit to center
                cv2.putText(img=board_img, text=digit, org=(x, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                            color=(0, 0, 0), thickness=3)
    return board_img



def solve_board(img, debug=False):
    processed_orig = process_image(img)
    if debug:
        while True:
            cv2.imshow('processed', processed_orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    corners = extract_board_corners(processed_orig)
    if corners is None:
        return None
    board_img, trans_matrix = perspective_warp(img, corners)
    inv_trans_matrix = np.linalg.inv(trans_matrix)
    if debug:
        while True:
            cv2.imshow('boardimgq', board_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    grid_unsolved = extract_board(board_img)
    grid_solved = SudokuSolver.solve(grid_unsolved)
    solved_img = fill_solution(board_img, grid_unsolved, grid_solved)
    if debug:
        while True:
            cv2.imshow('boardimgq', solved_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    # unwarp solution back to original image
    unwarped = cv2.warpPerspective(solved_img, inv_trans_matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, corners.astype(np.int), 0)
    solved = cv2.add(img, unwarped)
    while True:
        cv2.imshow('done', solved)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return solved


sudoku = cv2.imread('sudoku3.jpeg')
solve_board(sudoku, True)
# real_board = np.array([[9, 0, 0, 0, 0, 0, 3, 0, 0],
#                        [0, 0, 0, 5, 2, 0, 0, 0, 8],
#                        [0, 0, 0, 7, 0, 0, 0, 0, 0],
#                        [3, 0, 1, 0, 0, 9, 0, 0, 0],
#                        [0, 0, 5, 3, 0, 8, 0, 0, 0],
#                        [0, 7, 6, 0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 7, 4, 0],
#                        [0, 2, 0, 0, 8, 0, 0, 6, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 1, 5]])