import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os


class MyTools:
    def genrate_sample():
        # 图像尺寸
        width, height = 50, 50

        # 字体列表
        fonts_dict = {"Arial": "C:\\Windows\\Fonts\\arial.ttf", "Calibri": "C:\\Windows\\Fonts\\calibri.ttf", "Consola": "C:\\Windows\\Fonts\\consola.ttf", "Candara": "C:\\Windows\\Fonts\\candara.ttf", "Verdana": "C:\\Windows\\Fonts\\verdana.ttf"}

        font_size = 40
        center = (25, 25)
        for i in range(0, 10):
            for filename in os.listdir("C:\\Windows\\Fonts\\"):
                if os.path.splitext(filename)[-1] == ".ttf":
                    font_name = os.path.splitext(filename)[0]
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


MyTools.genrate_sample()
