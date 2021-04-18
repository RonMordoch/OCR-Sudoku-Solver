import cv2
import numpy as np
from ImageProcessor import solve_board
from SudokuSolver import solve


class Camera:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for main camera
        self._frame = None

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    def run(self):
        while True:
            ret, self._frame = self.cap.read()
            solved = solve_board(self._frame)
            # frame = solved if solved is not None else self._frame
            # cv2.imshow('AR Sudoku Solver', frame)
            cv2.imshow("main", self._frame)
            if solved is not None:
                cv2.imshow("board", solved)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
#
# c = Camera()
# c.run()
