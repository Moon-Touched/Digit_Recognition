import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os


class MyTools:
    def genrate_train_sample():
        """
        创建训练用的样本，获取windows/Fonts文件夹下所有.ttf文件，每个字体将数字打印在50×50的图片中心
        之后裁剪，resize为20×30的样本，文件命名{数字}_{字体}_{40}.png（40字号）

        ！！！待完善，同时使用opencv 高斯模糊，生成模糊的样本

        ！！！！系统字体可能有一些不包含数字，输出的图片无数字，需提前排除或输出后人工删除
        """
        # 图像尺寸
        width, height = 50, 50

        # 字体列表
        # fonts_dict = {"Arial": "C:\\Windows\\Fonts\\arial.ttf", "Calibri": "C:\\Windows\\Fonts\\calibri.ttf", "Consola": "C:\\Windows\\Fonts\\consola.ttf", "Candara": "C:\\Windows\\Fonts\\candara.ttf", "Verdana": "C:\\Windows\\Fonts\\verdana.ttf"}

        font_size = 40
        center = (25, 25)
        for i in range(0, 10):
            for filename in os.listdir("C:\\Windows\\Fonts\\"):
                if os.path.splitext(filename)[-1] == ".ttf":
                    font_name = os.path.splitext(filename)[0]
                    if font_name in ["wingding","webdings","symbol","segmdl2","holomdl2","marlett"]:
                        continue

                    font_path = "C:\\Windows\\Fonts\\" + filename
                    # 创建空白PIL图像并添加文本
                    pil_img = Image.new("RGB", (width, height), (0, 0, 0))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype(font_path, font_size)

                    # 应用文本
                    draw.text(center, str(i), font=font, fill=(255, 255, 255), anchor="mm")

                    # 转换为OpenCV图像
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

                    # 应用二值化
                    _, binary_img = cv2.threshold(cv_img, 50, 255, cv2.THRESH_BINARY)

                    # 找到轮廓
                    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    largest_contour = max(contours, key=cv2.contourArea)
                    # 轮廓尺寸
                    column, row, w, h = cv2.boundingRect(largest_contour)

                    # 保存图片
                    binary_img = binary_img[row : row + h, column : column + w]
                    binary_img = cv2.resize(binary_img, (20, 30))
                    cv2.imwrite(f"./train sample/{i}_{font_name}_{font_size}.png", binary_img)
                    cv2.imwrite(f"./train sample/{i}_{font_name}_{font_size}_blurred.png", cv2.GaussianBlur(binary_img, (5, 5), 0))

    def genrate_test_sample():
        """
        使用两个数独作为例子，裁切出数字用作测试用例，并输出文件，数字有重复，需人工筛选
        """
        file_list = ["example1.png", "example2.png"]
        # 预处理
        for file_name in file_list:
            image = cv2.imread(file_name)
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
            blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
            edged_image = cv2.Canny(blurred_image, 30, 150)
            contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                column, row, width, height = cv2.boundingRect(c)
                i = row // box_size
                j = column // box_size
                sub_boards = cv2.resize(binary_image[row : row + height, column : column + width], (20, 30))
                cv2.imwrite(f"./test sample/{i},{j}_{file_name}.png", sub_boards)
                    # cv2.imshow("b", box)
                    # cv2.waitKey(0)
    
    def genrate_test_sample2():
        """
        使用两个数独作为例子，裁切出数字用作测试用例，并输出文件，数字有重复，需人工筛选
        """
        file_list = ["example1.png", "example2.png"]
        # 预处理
        for file_name in file_list:
            image = cv2.imread(file_name)
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
            blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
            edged_image = cv2.Canny(blurred_image, 30, 150)
            contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                column, row, width, height = cv2.boundingRect(c)
                i = row // box_size
                j = column // box_size
                sub_boards = cv2.resize(binary_image[row : row + height, column : column + width], (20, 30))
                cv2.imwrite(f"./test sample/{i},{j}_{file_name}.png", sub_boards)

MyTools.genrate_test_sample2()
