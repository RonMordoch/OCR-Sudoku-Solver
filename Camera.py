import cv2
from ImageProcessor import ImageProcessor


class Camera:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for main camera
        self._frame = None
        self._img_processor = ImageProcessor()

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    def run(self):
        while True:
            ret, self._frame = self.cap.read()
            processed = self._img_processor.process_image(self._frame)
            # cv2.imshow('AR Sudoku Solver', self._frame)
            cnt = self._img_processor.extract_grid(processed)
            if cnt is not None and cv2.contourArea(cnt) > 10000:
                cv2.drawContours(self._frame, [cnt], 0, (0, ImageProcessor.MAX_INTENSITY, 0), 3)
            cv2.imshow('Original', self._frame)
            cv2.imshow('Processed', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


c = Camera()
c.run()
