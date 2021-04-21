# OCR-Sudoku-Solver


OCR Sudoku Solver.

Receive an input image:
![input](images/inputs/sudoku.jpeg)

Apply image processing to make board extraction easier:
![processed](images/processed.jpeg)

Extract the board from original image and apply perspective warp to obtain top-down view:
![board_img](images/board.jpeg)

Extract the digits and solve the board if found:
![board_solved](images/board_solved.jpeg)

Unwarp back the image:
![unwarp](images/unwarp.jpeg)

Fill with black the area in the original image where the board resides in:
![orig_filled_black](images/orig_filled_black.jpeg)

Paste the unwarped solution:
![final](images/final.jpeg)


## Sources:
* https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
* https://stackoverflow.com/questions/57636399/how-to-detect-sudoku-grid-board-in-opencv
* https://norvig.com/sudoku.html