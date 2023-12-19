import cv2
import os
import numpy as np
import tensorflow as tf


class Solver:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.board = np.zeros([9, 9])
        self.solved_board = np.zeros([9, 9])
        self.done = False

    def cut_board(self):
        """根据数独截图，将9宫格内的文字截取，缩放为30行，20列图像，存储在新矩阵对应位置

        Returns:
            sub_boards: 9×9 numpy array，每个矩阵元素即为相应9宫格的30行，20列图像，已归一化（/255）
        """
        image = cv2.imread(self.filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

        # 裁掉边框外的无用像素
        blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
        edged_image = cv2.Canny(blurred_image, 30, 150)
        contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        column, row, width, height = cv2.boundingRect(largest_contour)
        binary_image = binary_image[row : row + height, column : column + width]
        box_size = height // 9

        # 获取线条
        lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)
        # 在原图上绘制抹去检测到的线条
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(binary_image, (x1, y1), (x2, y2), 0, 5)

        # 检测文字边界并裁剪填入sub_board
        sub_boards = np.zeros((9, 9, 30, 20), dtype=np.uint8)
        blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
        edged_image = cv2.Canny(blurred_image, 30, 150)
        contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            column, row, width, height = cv2.boundingRect(c)
            i = row // box_size
            j = column // box_size
            sub_boards[i, j] = cv2.resize(binary_image[row : row + height, column : column + width], (20, 30)) / 255

        return sub_boards

    def match(self, sub_boards):
        model = tf.keras.models.load_model("digit_model",compile=False)
        for i in range(9):
            for j in range(9):
                if np.sum(sub_boards[i, j]) > 1:
                    res = model(np.array([sub_boards[i, j]]))
                    self.board[i, j] = res.numpy()[0].argmax()
        self.solved_board = self.board.copy()
        # W = tf.Variable(np.load("W.npy"))
        # V = tf.Variable(np.load("V.npy"))

        # b1 = tf.Variable(np.load("b1.npy"))
        # b2 = tf.Variable(np.load("b2.npy"))
        # for i in range(9):
        #     for j in range(9):
        #         if np.sum(sub_boards[i, j]) > 1:
        #             training_sample = tf.Variable(sub_boards[i, j].flatten().astype(np.float32))
        #             training_sample = tf.reshape(training_sample, [1, -1])
        #             h = tf.nn.sigmoid(tf.matmul(training_sample, W) + b1)
        #             logits = tf.matmul(h, V) + b2
        #             y = tf.nn.softmax(logits)
        #             self.board[i, j] = y.numpy().argmax()
        # self.solved_board = self.board.copy()

    def backtrack(self, index):
        if index == 81:
            self.done = True
            return
        if not self.done:
            i = index // 9
            j = index % 9
            if self.solved_board[i][j] == 0:
                for n in range(1, 10):
                    if (not n in self.solved_board[i]) and (not n in [self.solved_board[x][j] for x in range(9)]) and (not n in [self.solved_board[x][y] for x in range(3 * (i // 3), 3 * (i // 3) + 3) for y in range(3 * (j // 3), 3 * (j // 3) + 3)]):
                        self.solved_board[i][j] = n
                        self.backtrack(index + 1)
                        if self.done:
                            return
                        self.solved_board[i][j] = 0
            else:
                self.backtrack(index + 1)
        return

    def solve(self):
        sub_boards = self.cut_board()
        self.match(sub_boards)
        self.backtrack(0)
        print(self.board)
        print(self.solved_board)


s = Solver("./Sudoku/board/example3.png")
s.solve()
